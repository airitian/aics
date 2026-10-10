#!/usr/bin/env python
"""把 PostgreSQL 里已有的 chunks 重新算 embedding，回填进自建 Qdrant 的两个集合。

背景
----
AICS 从 Qdrant Cloud 迁到自建 Qdrant 后，关系数据（19 表 6042 行）已迁完并校验，
但向量集合 ``aics_chunks``（legacy）与 ``aics_chunks_v2``（llama）都是 0 点。
PostgreSQL 的 ``chunks`` 表文本完好无损，所以**不需要重新上传原始文件、不需要重新
解析文档**——直接拿现有 chunk 重算 embedding 即可，切分结果与切换前完全一致。

与生产行为的等价性（逐条对齐源码，改了任何一条都会让检索语义漂移）
------------------------------------------------------------------
1. 读取范围：``chunks WHERE enabled = true``，按 ``doc_id, seq`` 排序。
2. 跳过规则：命中 ``app.rag._ID_ROW_RE``（订单编号/运单号/SKU 等"标签: 长数字"）的
   chunk **不建向量**——它们是语义检索的万金油噪声，生产入库时就跳过
   （见 app/rag.py:262 与 app/rag_llama.py:257）。
3. embedding 输入是**剥掉元数据前缀**的文本（``app.rag._strip_meta``），
   而写入的 node.text 是**原文**（含「【…】《…》」前缀）。这个不一致是有意为之：
   前缀词面会稀释相关性（实测同一条内容裸句 rerank 0.33，带前缀跌到 0.04），
   但 SQLite/LLM 上下文必须保留完整文本做来源归属。
4. Node 构造复用 ``app.rag_llama._make_node``，保证 point id、metadata 四个键、
   ``relationships[SOURCE]``（决定 payload 的 document_id/ref_doc_id，缺了它
   按文档删除就失效）与生产完全一致。
5. embedding 走 ``app.embedding.embed_texts``（内部按 EMBED_BATCH 分批、
   带退避重试与熔断），不裸调 HTTP。

两个集合的差异
--------------
* llama ``aics_chunks_v2``：``llama_vector_store().add(nodes)``，payload 平铺，
  含 ``_node_content``（可还原 node id 与原文）。
* legacy ``aics_chunks``：``build_vector_store(db, dim).upsert([VectorItem...])``，
  payload 只有 ``chunk_id/tenant_id/kb_id/doc_id`` 四字段，不含 text。
* 归一化：legacy 侧 ``QdrantVectorStore.upsert`` 内部第 257 行已经做过
  ``l2_normalize``，这里**不要重复归一化**（对单位向量重复归一化是幂等的，但会
  误导读者）；llama 侧不需要归一化。

用法
----
    # 1) 先干跑：只读库统计，不打 embedding、不写 Qdrant
    python rebuild_vectors_from_pg.py --dry-run

    # 2) 小批量验证链路
    python rebuild_vectors_from_pg.py --limit 8

    # 3) 全量（崩了就带 --resume 续跑，已写入的点会跳过，不重复烧额度）
    python rebuild_vectors_from_pg.py --resume

退出码：全部成功 0；有任一 chunk 失败 1；参数/环境校验不通过 2。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.embedding import embed_texts
from app.rag import _ID_ROW_RE, _strip_meta
from app.rag_llama import _make_node, _node_id, llama_vector_store
from app.vectorstore import VectorItem, build_vector_store

logger = logging.getLogger("aics.rebuild_vectors")

# 读库语句：enabled=false 的 chunk 生产上也不进向量库，必须保持一致
_SELECT_CHUNKS = text(
    "SELECT id, tenant_id, kb_id, doc_id, seq, text "
    "FROM chunks WHERE enabled = true "
    "ORDER BY doc_id, seq"
)


# --------------------------------------------------------------------------- #
# 数据载体
# --------------------------------------------------------------------------- #
@dataclass
class ChunkRow:
    """一条待回填的 chunk。seq 统一转 int：metadata 里的类型必须和生产一致。"""

    id: str
    tenant_id: str
    kb_id: str
    doc_id: str
    seq: int
    text: str

    @property
    def point_id(self) -> str:
        """Qdrant point id（与 rag_llama._node_id 同一套算法）。"""
        return _node_id(self.id)


@dataclass
class Stats:
    """累计计数器。skip 分两类：生产规则跳过（噪声块）与续跑跳过（已存在）。"""

    ok: int = 0
    fail: int = 0
    skipped_noise: int = 0
    skipped_existing: int = 0
    tokens: int = 0


# --------------------------------------------------------------------------- #
# 读库与计划
# --------------------------------------------------------------------------- #
def load_chunks(db: Session, limit: int | None) -> list[ChunkRow]:
    """按 doc_id, seq 读出所有 enabled 的 chunk。limit 只影响处理条数，不改排序。"""
    rows = db.execute(_SELECT_CHUNKS).mappings().all()
    out: list[ChunkRow] = []
    for r in rows:
        out.append(
            ChunkRow(
                id=str(r["id"]),
                tenant_id=str(r["tenant_id"]),
                kb_id=str(r["kb_id"]),
                doc_id=str(r["doc_id"]),
                seq=int(r["seq"] or 0),
                text=str(r["text"] or ""),
            )
        )
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out


def split_noise(rows: Sequence[ChunkRow]) -> tuple[list[ChunkRow], list[ChunkRow]]:
    """按生产规则把噪声块（标识符行）挑出来。

    这类块对任何提问都有 ~0.5 的上下文向量分，会把真实内容挤出召回池；
    它们只服务"用户粘贴单号"的精确查询（走 SQLite 全文，不走向量）。
    """
    keep: list[ChunkRow] = []
    noise: list[ChunkRow] = []
    for row in rows:
        if _ID_ROW_RE.search(row.text):
            noise.append(row)
        else:
            keep.append(row)
    return keep, noise


def print_plan(rows: Sequence[ChunkRow], keep: Sequence[ChunkRow], noise: Sequence[ChunkRow]) -> None:
    """打印按文档分组的计划，便于人工核对数量。"""
    logger.info("=" * 72)
    logger.info("PG chunks 表 enabled=true 共 %d 行", len(rows))
    per_doc: dict[str, int] = {}
    per_doc_keep: dict[str, int] = {}
    for r in rows:
        per_doc[r.doc_id] = per_doc.get(r.doc_id, 0) + 1
    for r in keep:
        per_doc_keep[r.doc_id] = per_doc_keep.get(r.doc_id, 0) + 1
    for doc_id, n in sorted(per_doc.items(), key=lambda kv: -kv[1]):
        logger.info(
            "  doc %s  总 %4d 行 → 应建向量 %4d（跳过噪声 %d）",
            doc_id, n, per_doc_keep.get(doc_id, 0), n - per_doc_keep.get(doc_id, 0),
        )
    logger.info("合计：读 %d 行，命中标识符规则跳过 %d 行，应建向量 %d 条", len(rows), len(noise), len(keep))
    logger.info("=" * 72)


# --------------------------------------------------------------------------- #
# 续跑：查已存在的 point id
# --------------------------------------------------------------------------- #
def make_qdrant_client() -> Any:
    """自建 Qdrant 的裸客户端（只用于 retrieve/count/get_collection，不写数据）。"""
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
        prefer_grpc=False,
    )


def existing_ids(client: Any, collection: str, point_ids: Sequence[str], probe: int = 200) -> set[str]:
    """批量判定哪些 point id 已存在。

    为什么用 retrieve 而不是 count：count 只能给出总数，无法区分"哪些已完成"，
    而续跑必须精确到 id——重算已完成的点等于白烧 embedding 额度。
    """
    found: set[str] = set()
    for i in range(0, len(point_ids), probe):
        batch = list(point_ids[i : i + probe])
        try:
            got = client.retrieve(
                collection_name=collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 - 集合不存在/网络抖动都当"未存在"
            logger.warning("查询集合 %s 已有点失败（视为全部未存在）：%s", collection, exc)
            return found
        found.update(str(p.id) for p in got)
    return found


# --------------------------------------------------------------------------- #
# embedding（外层再包一层退避，embed_texts 内部已有重试 + 熔断）
# --------------------------------------------------------------------------- #
async def embed_with_retry(texts: list[str], *, attempts: int = 3, base_delay: float = 2.0) -> list[list[float]]:
    """调 embed_texts，失败按 2/4/8 秒退避重试。

    外层重试的理由：embed_texts 的熔断是对**整个进程**的（连续失败 3 次就开闸），
    全量 286 条里偶发一次 429 不该把后面的批次一起带进冷却。这里把单次失败
    消化在本批内，只有连续 attempts 次都失败才计 fail 并跳过本批。
    """
    last: BaseException | None = None
    for i in range(attempts):
        try:
            res = await embed_texts(texts)
            if len(res.vectors) != len(texts):
                raise RuntimeError(f"向量条数与请求不一致（{len(res.vectors)}/{len(texts)}）")
            if res.dim != settings.embed_dim:
                raise RuntimeError(f"向量维度 {res.dim} 与 EMBED_DIM {settings.embed_dim} 不一致")
            return res.vectors
        except Exception as exc:  # noqa: BLE001 - 429/5xx/网络抖动都要重试
            last = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            logger.warning("embedding 第 %d/%d 次失败（%.1fs 后重试）：%s", i + 1, attempts, delay, exc)
            await asyncio.sleep(delay)
    raise RuntimeError(f"embedding 连续 {attempts} 次失败：{last}") from last


# --------------------------------------------------------------------------- #
# 写入
# --------------------------------------------------------------------------- #
def write_llama(rows: Sequence[ChunkRow], vectors: Sequence[Sequence[float]]) -> int:
    """写 llama 集合 aics_chunks_v2。节点构造完全复用生产的 _make_node。"""
    nodes = [
        _make_node(
            row.id,
            row.text,   # 原文：含「【…】《…》」前缀，不做任何加工
            tenant_id=row.tenant_id,
            kb_id=row.kb_id,
            doc_id=row.doc_id,
            seq=row.seq,
            vector=list(vec),
        )
        for row, vec in zip(rows, vectors)
    ]
    if not nodes:
        return 0
    llama_vector_store().add(nodes)
    return len(nodes)


def write_legacy(store: Any, rows: Sequence[ChunkRow], vectors: Sequence[Sequence[float]]) -> int:
    """写 legacy 集合 aics_chunks。

    注意：l2_normalize 由 QdrantVectorStore.upsert 内部完成（qdrant_store.py:257），
    这里不再重复归一化；payload 只有 chunk_id/tenant_id/kb_id/doc_id 四个字段。
    """
    items = [
        VectorItem(
            id=row.id,
            tenant_id=row.tenant_id,
            kb_id=row.kb_id,
            doc_id=row.doc_id,
            text=row.text,
            vector=list(vec),
        )
        for row, vec in zip(rows, vectors)
    ]
    if not items:
        return 0
    store.upsert(items)
    return len(items)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 PostgreSQL 的 chunks 表重算 embedding 并回填自建 Qdrant 的两个集合",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="只读库统计，不调 embedding、不写 Qdrant")
    p.add_argument("--limit", type=int, default=None, help="只处理排序后的前 N 条（小批量验证链路用）")
    p.add_argument("--resume", action="store_true", help="跳过两个集合里都已存在的点，避免重复烧 embedding 额度")
    p.add_argument("--batch", type=int, default=settings.embed_batch, help="每批条数（与 EMBED_BATCH 对齐）")
    p.add_argument("--sleep", type=float, default=0.5, help="每批之间的间隔秒数（Gitee AI 会 429）")
    p.add_argument(
        "--only",
        choices=("both", "llama", "legacy"),
        default="both",
        help="只写某一个集合（默认两个都写）",
    )
    p.add_argument("--log-level", default="INFO", help="日志级别")
    return p.parse_args(argv)


def preflight() -> int:
    """环境校验：后端必须是 qdrant，embedding 必须是真实 API 且维度 1024。"""
    if settings.vector_backend != "qdrant":
        logger.error("VECTOR_BACKEND=%s，本脚本只在 qdrant 后端下有意义", settings.vector_backend)
        return 2
    if not settings.qdrant_url:
        logger.error("QDRANT_URL 未配置")
        return 2
    if settings.embed_provider != "api":
        logger.error("EMBED_PROVIDER=%s，本地哈希向量没有真实语义，不能回填生产集合", settings.embed_provider)
        return 2
    logger.info(
        "配置：QDRANT_URL=%s | legacy=%s | llama=%s | model=%s dim=%s batch=%s",
        settings.qdrant_url, settings.qdrant_collection,
        settings.qdrant_collection_llama, settings.embed_model,
        settings.embed_dim, settings.embed_batch,
    )
    return 0


async def run(args: argparse.Namespace) -> int:
    rc = preflight()
    if rc:
        return rc

    started = time.time()
    db = SessionLocal()
    try:
        rows = load_chunks(db, args.limit)
    finally:
        db.close()

    keep, noise = split_noise(rows)
    print_plan(rows, keep, noise)

    if args.dry_run:
        logger.info("--dry-run：到此为止，未调用 embedding、未写入 Qdrant")
        return 0

    if not keep:
        logger.warning("没有需要建向量的 chunk，直接结束")
        return 0

    stats = Stats()
    stats.skipped_noise = len(noise)

    client = make_qdrant_client()
    # legacy VectorStore 构造期要做集合/维度校验，需要一个独立 Session
    store_db: Session | None = SessionLocal() if args.only in ("both", "legacy") else None
    store = build_vector_store(store_db, settings.embed_dim) if store_db is not None else None

    # 续跑：先把已存在的点一次性算出来（两个集合都要有，缺一侧就重算，避免不一致）
    done_llama: set[str] = set()
    done_legacy: set[str] = set()
    if args.resume:
        ids = [r.point_id for r in keep]
        if args.only in ("both", "llama"):
            done_llama = existing_ids(client, settings.qdrant_collection_llama, ids)
        if args.only in ("both", "legacy"):
            done_legacy = existing_ids(client, settings.qdrant_collection, ids)
        logger.info(
            "--resume：%s 已有 %d 点，%s 已有 %d 点",
            settings.qdrant_collection_llama, len(done_llama),
            settings.qdrant_collection, len(done_legacy),
        )

    def is_done(row: ChunkRow) -> bool:
        """判定该点是否已无需重算。

        both 模式要求**两个集合都有**才跳过——只补一侧会让两个集合不一致，
        而 legacy 是回滚路径，必须随时可用。单集合模式只按本侧判定。
        """
        if not args.resume:
            return False
        if args.only == "llama":
            return row.point_id in done_llama
        if args.only == "legacy":
            return row.point_id in done_legacy
        return row.point_id in done_llama and row.point_id in done_legacy

    pending = [r for r in keep if not is_done(r)]
    stats.skipped_existing = len(keep) - len(pending)

    total = len(pending)
    logger.info("待处理 %d 条（噪声跳过 %d，续跑跳过 %d）", total, stats.skipped_noise, stats.skipped_existing)

    batch_size = max(1, args.batch)
    for start in range(0, total, batch_size):
        batch = pending[start : start + batch_size]
        done = start + len(batch)
        try:
            # embedding 用剥掉元数据前缀的正文；写入的 text 用原文
            vectors = await embed_with_retry([_strip_meta(r.text) for r in batch])
        except Exception as exc:  # noqa: BLE001 - 本批整体失败，记 fail 后继续下一批
            logger.error("[%d/%d] 本批 embedding 失败，跳过 %d 条：%s", done, total, len(batch), exc)
            stats.fail += len(batch)
            time.sleep(args.sleep)
            continue

        try:
            if args.only in ("both", "llama"):
                write_llama(batch, vectors)
            if args.only in ("both", "legacy") and store is not None:
                write_legacy(store, batch, vectors)
        except Exception as exc:  # noqa: BLE001 - 写入失败同样不让整轮崩掉，便于 --resume 续跑
            logger.error("[%d/%d] 写入 Qdrant 失败，本批 %d 条记为失败：%s", done, total, len(batch), exc)
            stats.fail += len(batch)
            time.sleep(args.sleep)
            continue

        stats.ok += len(batch)
        logger.info(
            "[%d/%d] ok=%d skip_noise=%d skip_exist=%d fail=%d",
            done, total, stats.ok, stats.skipped_noise, stats.skipped_existing, stats.fail,
        )
        if args.sleep > 0 and done < total:
            time.sleep(args.sleep)

    elapsed = time.time() - started
    logger.info("=" * 72)
    logger.info("耗时 %.1fs | 成功 %d | 噪声跳过 %d | 续跑跳过 %d | 失败 %d",
                elapsed, stats.ok, stats.skipped_noise, stats.skipped_existing, stats.fail)
    try:
        for coll in (settings.qdrant_collection_llama, settings.qdrant_collection):
            n = client.count(collection_name=coll, exact=True).count
            logger.info("集合 %s 当前点数 = %d", coll, n)
    except Exception as exc:  # noqa: BLE001 - 收尾统计失败不影响退出码判定
        logger.warning("读取集合点数失败：%s", exc)
    logger.info("=" * 72)
    if store_db is not None:
        store_db.close()
    return 1 if stats.fail else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # qdrant_client / httpx 的 INFO 请求日志会把进度刷没，压到 WARNING
    for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3", "jieba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.warning("已中断；重跑加 --resume 可跳过已完成的点")
        return 1


if __name__ == "__main__":
    sys.exit(main())
