"""Qdrant 真机验证：证明多租户隔离在云端向量库上真正生效。

与 smoke_test.py 的区别：
- smoke_test.py 打 HTTP，验证端到端链路；
- 本脚本直接压向量层，用「**同一个向量、只换 tenant_id 过滤**」来证伪
  「查不到是因为数据本来就是空的」这一可能。

用法：
    python verify_qdrant.py            # 跑完后自动清理测试租户
    python verify_qdrant.py --keep     # 保留测试数据供人工在 Qdrant 控制台查看

前置：.env 里 VECTOR_BACKEND=qdrant 且 QDRANT_URL / QDRANT_API_KEY 已填。
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import Document, KnowledgeBase
from app.provision import create_tenant
from app.rag import index_document, retrieve
from app.teardown import purge_tenant_data
from app.vectorstore import VectorItem, build_vector_store
from app.vectorstore.base import VectorStoreUnavailable

A_SECRET = "QDRANT-ONLY-ALPHA-7781"
B_SECRET = "QDRANT-ONLY-BETA-9924"

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def banner(text: str) -> None:
    print(f"\n{text}")
    print("-" * 68)


def _default_kb(db, tenant_id: str) -> str:
    return db.execute(
        select(KnowledgeBase.id).where(KnowledgeBase.tenant_id == tenant_id)
    ).scalars().first()


def _add_doc(db, tenant_id: str, kb_id: str, text: str) -> str:
    from app.utils import new_id

    doc = Document(
        id=new_id(),
        tenant_id=tenant_id,
        kb_id=kb_id,
        filename="verify.txt",
        ext=".txt",
        size_bytes=len(text.encode("utf-8")),
        status="ready",
    )
    db.add(doc)
    db.flush()
    return doc.id


def _purge_tenant(db, store, tenant_id: str) -> None:
    """复用生产同一份销毁逻辑（app/teardown.py），避免测试与线上删除顺序漂移。"""
    purge_tenant_data(db, tenant_id)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留测试数据不清理")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="先删掉整个集合再重建（会清空已有向量），保证可重复验证",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="确认删除：集合非空、或 ENV 不是 dev 时，--reset 默认拒绝执行",
    )
    args = ap.parse_args()

    banner("0. 当前配置")
    print(f"  VECTOR_BACKEND    = {settings.vector_backend}")
    print(f"  QDRANT_URL        = {settings.qdrant_url}")
    print(f"  QDRANT_COLLECTION = {settings.qdrant_collection}")
    print(f"  EMBED_PROVIDER    = {settings.embed_provider}  (dim={settings.embed_dim})")
    if settings.vector_backend != "qdrant":
        print("\n  !! 当前不是 qdrant 后端，本脚本无意义。请在 .env 设 VECTOR_BACKEND=qdrant")
        return 2

    if args.reset:
        import qdrant_client

        from app.vectorstore import qdrant_store as _qs

        coll = settings.qdrant_collection
        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=20
        )

        # 删除集合不可逆，所以这里一律 fail-closed：
        # 不是 dev/test 环境、或读不到集合内容（无法确认删什么），默认都不删。
        if settings.env not in ("dev", "test") and not args.force:
            print(
                f"\n  !! 拒绝执行 --reset：当前 ENV={settings.env}，删除集合 {coll} 不可逆。\n"
                f"     确认要删请显式加 --force。"
            )
            return 2

        try:
            points: int | None = client.get_collection(coll).points_count
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:120]}"
            if not args.force:
                print(
                    f"\n  !! 拒绝执行 --reset：读不到集合 {coll} 的信息（{detail}）。\n"
                    f"     读不到内容就不动它——先确认网络/集群状态；确实要强删可加 --force。"
                )
                return 2
            points = None  # 未知：仅在 --force 下会走到这里，随后直接删除

        if points and not args.force:
            print(
                f"\n  !! 集合 {coll} 内仍有 {points} 个点，--reset 会全部删除且无法恢复。\n"
                f"     如果这确实是测试数据，请显式加 --force 重跑。"
            )
            return 2

        print(f"\n  --reset：删除集合并重建 {coll}（删除前点数={points}）")
        try:
            client.delete_collection(coll)
        except Exception as exc:
            print(f"\n  !! 删除集合失败：{type(exc).__name__}: {str(exc)[:140]}")
            print("     集合未被改动，可先修好网络再重试。")
            return 3
        # 让缓存失效：既包括「集合已就绪」标记，也包括可能已经打开的熔断器
        _qs._READY.clear()
        _qs._BREAKERS.clear()

    init_db()
    db = SessionLocal()
    t_a = t_b = None

    try:
        banner("1. 向量库连通性 / 集合就绪")
        try:
            store = build_vector_store(db)
            health = store.healthcheck()
        except VectorStoreUnavailable as exc:
            print(f"  !! 向量库不可达：{exc.detail or exc.message}")
            print("     排查顺序：")
            print("       1) 本机出口网络 / 代理是否放行该域名（TLS 被重置时典型症状是 UNEXPECTED_EOF）")
            print("       2) QDRANT_URL 与 QDRANT_API_KEY 是否正确、API Key 是否已失效")
            print("       3) Qdrant Cloud 控制台里集群是否处于运行状态（免费版可能被暂停）")
            return 3
        print(f"  health = {health}")
        check("Qdrant 可达且集合已就绪", bool(health.get("ok")), str(health.get("error", "")))
        check(
            "集合维度与 EMBED_DIM 一致",
            health.get("dim") == settings.embed_dim,
            f"集合={health.get('dim')} 配置={settings.embed_dim}",
        )
        before = health.get("points") or 0
        print(f"  入库前集合点数 = {before}")
        if before:
            print(f"  提示：集合中已有 {before} 个点（可能是上次异常中断的残留）。"
                  f"要干净复现场景可加 --reset。")

        banner("2. 开通两个临时租户，各写一条独占知识")
        ta, _ = create_tenant(db, name=f"验证租户A{settings.embed_dim}", admin_email="vt-a@verify.local",
                              admin_password="verify12345")
        tb, _ = create_tenant(db, name=f"验证租户B{settings.embed_dim}", admin_email="vt-b@verify.local",
                              admin_password="verify12345")
        t_a, t_b = ta.id, tb.id
        kb_a, kb_b = _default_kb(db, t_a), _default_kb(db, t_b)
        db.commit()
        check("两个租户各自拿到默认知识库", bool(kb_a and kb_b))
        check("两租户知识库互不相同", kb_a != kb_b)

        text_a = f"本公司的内部结算密钥是 {A_SECRET}，仅限 A 租户内部使用，对外一律保密。"
        text_b = f"本公司的内部结算密钥是 {B_SECRET}，仅限 B 租户内部使用，对外一律保密。"
        doc_a = _add_doc(db, t_a, kb_a, text_a)
        doc_b = _add_doc(db, t_b, kb_b, text_b)
        await index_document(db, tenant_id=t_a, kb_id=kb_a, doc_id=doc_a, text=text_a)
        await index_document(db, tenant_id=t_b, kb_id=kb_b, doc_id=doc_b, text=text_b)
        db.commit()

        after = store.healthcheck().get("points") or 0
        print(f"  入库后集合点数 = {after}（+{after - before}）")
        check("向量已真实写入 Qdrant", after > before, f"{before} -> {after}")

        banner("3. 检索隔离（业务视角：带 kb 范围）")
        hits_a, _ = await retrieve(db, tenant_id=t_a, kb_ids=[kb_a], query="内部结算密钥")
        hits_b, _ = await retrieve(db, tenant_id=t_b, kb_ids=[kb_b], query="内部结算密钥")
        got_a = " ".join(h.text for h in hits_a)
        got_b = " ".join(h.text for h in hits_b)
        check("A 能检索到自己的机密", A_SECRET in got_a, f"命中 {len(hits_a)} 条")
        check("A 检索不到 B 的机密", B_SECRET not in got_a, got_a[:80])
        check("B 能检索到自己的机密", B_SECRET in got_b, f"命中 {len(hits_b)} 条")
        check("B 检索不到 A 的机密", A_SECRET not in got_b, got_b[:80])

        banner("4. 硬核证伪：同一个向量，只换租户过滤")
        # 设计要点：A 与 B 的知识**故意写得高度相似**（只差机密串），
        # 用的是本地哈希向量，所以 B 原文的向量与 A 的文本相似度也很高。
        # 这样一旦 payload 过滤失效，A 必然召回 B —— 排除「查不到是因为库里本来没有」。
        from app.embedding import embed_query
        from app.models import Chunk

        a_chunks = set(
            db.execute(select(Chunk.id).where(Chunk.tenant_id == t_a)).scalars().all()
        )
        b_chunks = set(
            db.execute(select(Chunk.id).where(Chunk.tenant_id == t_b)).scalars().all()
        )

        probe_vec, _ = await embed_query(text_b)
        raw_a = store.search(t_a, probe_vec, top_k=10, min_score=0.0)
        raw_b = store.search(t_b, probe_vec, top_k=10, min_score=0.0)
        ids_a = {h.chunk_id for h in raw_a}
        ids_b = {h.chunk_id for h in raw_b}
        print(f"  用 B 的原文向量检索：tenant_A 命中 {len(ids_a)} 条，tenant_B 命中 {len(ids_b)} 条")

        # 不加任何租户过滤，直接问 Qdrant「这个向量最近的是谁」。
        # 只有这一步能证明过滤是**承重的**：最相似的那条确实属于 B。
        import qdrant_client

        raw_client = qdrant_client.QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=20
        )
        top = raw_client.query_points(
            collection_name=settings.qdrant_collection,
            query=probe_vec,
            limit=3,
            with_payload=True,
        ).points
        top_owners = [(p.payload.get("tenant_id"), round(float(p.score), 4)) for p in top]
        print(f"  不加过滤时 Qdrant 的最近邻（tenant_id, score）= {top_owners}")
        check(
            "不加过滤时最近邻里存在 B 的分片（证明向量确实指向 B）",
            any(p.payload.get("tenant_id") == t_b for p in top),
            str(top_owners),
        )
        check(
            "加 A 的过滤后，返回的分片全部属于 A（B 的一条都没混进来）",
            ids_a.issubset(a_chunks) and not (ids_a & b_chunks),
            f"A 命中含 B 分片：{ids_a & b_chunks}",
        )
        check("加 B 的过滤后能命中 B 自己的分片", bool(ids_b & b_chunks), f"命中 {ids_b}")

        banner("5. 写入侧隔离：伪造 tenant_id 的向量不会串租户")
        from app.utils import new_id

        forged_id = new_id()
        store.upsert([VectorItem(id=forged_id, tenant_id=t_a, kb_id=kb_a,
                                 doc_id=doc_a, text="伪造", vector=probe_vec)])
        seen_by_a = store.search(t_a, probe_vec, top_k=10, min_score=0.0)
        seen_by_b = store.search(t_b, probe_vec, top_k=10, min_score=0.0)
        check("写入 A 的向量只出现在 A 的结果里",
              any(h.chunk_id == forged_id for h in seen_by_a)
              and not any(h.chunk_id == forged_id for h in seen_by_b))
        store.delete_by_doc(t_a, doc_a)

        banner("6. 维度守卫：换 embedding 模型时必须炸得明白")
        from app.vectorstore.qdrant_store import DimensionMismatch, QdrantVectorStore

        wrong = settings.embed_dim + 128
        try:
            QdrantVectorStore(wrong)
            check("维度不匹配时应抛 DimensionMismatch", False, f"dim={wrong} 竟然通过了")
        except DimensionMismatch as exc:
            check("维度不匹配时抛出 DimensionMismatch 并给出可执行修法", True)
            for line in str(exc).splitlines():
                print(f"       {line}")
        except Exception as exc:  # pragma: no cover
            check("维度不匹配应抛 DimensionMismatch", False, f"{type(exc).__name__}: {exc}")

        banner("7. 清理")
        if args.keep:
            print("  --keep 指定，保留测试数据。可在 Qdrant 控制台按 payload.tenant_id 过滤查看：")
            print(f"    tenant_A={t_a}  kb_A={kb_a}")
            print(f"    tenant_B={t_b}  kb_B={kb_b}")
        else:
            _purge_tenant(db, store, t_a)
            _purge_tenant(db, store, t_b)
            db.commit()
            remain = store.healthcheck().get("points") or 0
            print(f"  清理后集合点数 = {remain}（原始为 {before}）")
            check("测试数据已清理干净", remain == before, f"{before} -> {remain}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    banner("结果")
    print(f"  通过 {PASS} 项，失败 {FAIL} 项")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
