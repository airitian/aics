"""RAG：切分、入库、检索。

隔离要点：
- 入库时每条 chunk 都带 tenant_id；
- 检索时先把候选集限制在本租户 + 本 AI 员工绑定的知识库，再由向量库二次过滤；
- **多库检索互相隔离**：每个绑定的知识库独立搜索、独立评分，各取保底候选后再合并
  （见 retrieve），确保每一份知识库的内容单独分开、互不挤占；
- 取回正文时**再查一次库并带 tenant_id 条件**，形成双保险。
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app import ratelimit
from app.config import settings
from app.embedding import embed_query, embed_texts
from app.models import Chunk, Document, EmployeeKbBinding
from app.rerank import enabled as rerank_enabled
from app.rerank import rerank as rerank_passages
from app.utils import new_id
from app.vectorstore import VectorItem, build_vector_store

logger = logging.getLogger("aics.rag")

_SENTENCE_END = "。！？!?；;\n"
# 次级切点：找不到句末时退而求其次。中文长行（引用块、表格说明）常常几百字才一个句号，
# 只认句末就只能在字數上限处硬切，切出来的下一片开头是半句话。
_SOFT_END = "，、,)）」】；; "
# 结构化块渲染出来的路径前缀，例如「【产品中心 > 无线网桥】\n」。
# 块被滑窗切成多片时，后续片必须补回前缀，否则第二片就丢了章节归属。
_PATH_PREFIX = re.compile(r"^【[^】]{1,200}】\n")

# 标识符行：表格解析块里"标签: 长数字"形态的行（订单编号/运单号/SKU 等）。
# 这类块对语义查询是万金油噪声（任何提问都有 0.5 上下文的向量分），
# 只在查询本身携带长数字串（用户粘贴单号）时才有语义检索价值。
_ID_ROW_RE = re.compile(
    r"(?:订单编号|运单号|物流单号|货号|商品编号|SKU\s*ID|Offer\s*ID|单号)\s*[：:]\s*\d{6,}"
)
_QUERY_HAS_LONG_ID = re.compile(r"\d{6,}")

# 售后意图与联系方式块：见 _service_supplement。
# 【售后服务】块词面与具体故障提问几乎零重叠（"驱动轮卡住了怎么修" vs
# "售后热线 400-xxx-xxxx"），向量与精排都竞争不过故障排查块；而模型受
# "严禁编造"约束，上下文里没有号码就不敢写。查询命中售后/维修类意图时，
# 从 SQLite 直捞该块兜底注入。
_SERVICE_QUERY_RE = re.compile(
    r"售后|维修|修理|故障|坏了|不能用|无法使用|客服|人工|热线|电话|报修|联系|修吗|怎么办"
)
_SERVICE_BLOCK_RE = re.compile(r"^【售后服务】")

# 块型/文件名元数据前缀（「【要点清单】《xxx.pdf》」「【产品中心 > 网桥】」）。
# 实测这类前缀会毒化打分：同一条内容裸句 rerank 0.33，带「【块型】《文件名》」
# 前缀跌到 0.04（元数据词面稀释相关性）。因此**向量与 rerank 一律用剥掉前缀
# 的纯正文**；SQLite 与 LLM 上下文仍保留完整文本（来源归属不丢）。
_META_PREFIX = re.compile(r"^(?:【[^】]{0,80}】|《[^》]{0,160}》|\s)+")


def _strip_meta(text: str) -> str:
    return _META_PREFIX.sub("", text or "", count=1) or text


@dataclass
class RetrievedChunk:
    chunk_id: str
    kb_id: str
    doc_id: str
    filename: str
    score: float
    text: str


@dataclass
class IndexResult:
    chunk_count: int = 0
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def _window(block: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    n = len(block)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # 在窗口后 30% 内找最近的句末，避免把句子切半；
            # 找不到句末时退到次级切点（逗号、顿号），至少保证不在词中间断开。
            search_from = start + int(size * 0.7)
            cut = -1
            for stops in (_SENTENCE_END, _SOFT_END):
                for i in range(end, search_from, -1):
                    if block[i - 1] in stops:
                        cut = i
                        break
                if cut > 0:
                    break
            if cut > 0:
                end = cut
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_text(
    text: str,
    *,
    size: int | None = None,
    overlap: int | None = None,
    min_fill: int | None = None,
) -> list[str]:
    """按段落切分；短段落会合并，但**不会一路并到 size 上限**。

    `min_fill` 决定了「并段的胃口」：缓冲区装到 min_fill 就不再并了，直接切一块。
    为什么不能像以前那样一直并到 size：语义模型对「一块里塞了多个无关主题」非常敏感。
    实测（bge-m3）：把 6 条客服 FAQ 并在一个 302 字的块里，
      直接问其中一条的折扣码相似度只有 0.51 —— 低于召回门槛，等于问了答不上来；
      而同一条独立成块时是 0.78，稳稳命中。
    并段的初衷只是避免「十几个字的碎块」，不是把整篇并成一个块。
    默认取 size//3：段落级粒度（多数段落自成一块），只在段落很短时才并。
    """
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    size = max(100, size)
    overlap = max(0, min(overlap, size // 2))
    fill = min_fill if min_fill is not None else size // 3
    fill = max(0, min(fill, size))

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    if not blocks:
        return []

    chunks: list[str] = []
    buf = ""
    for block in blocks:
        if len(block) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_window(block, size, overlap))
            continue
        if not buf:
            buf = block
        elif len(buf) < fill and len(buf) + len(block) + 1 <= size:
            buf = f"{buf}\n{block}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{block}" if tail else block
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def chunk_blocks(
    blocks: list[str],
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """解析层给出的**原子块**（表格记录）直接成块，不参与并段。

    为什么不走 chunk_text：表格记录本身已自足（带表名、列名、主键），
    再按 min_fill 并段会把多条无关记录塞进一块，重演「6 条 FAQ 并在一块、
    相似度 0.78→0.51 掉到召回门槛下」的老问题。只有超过 size 的块才做滑窗。
    """
    size = max(100, size or settings.chunk_size)
    overlap = max(0, min(overlap if overlap is not None else settings.chunk_overlap, size // 2))
    out: list[str] = []
    for block in blocks:
        block = (block or "").strip()
        if not block:
            continue
        if len(block) <= size:
            out.append(block)
            continue
        pieces = _window(block, size, overlap)
        m = _PATH_PREFIX.match(block)
        if m and len(pieces) > 1:
            prefix = m.group(0)
            pieces = [
                p if i == 0 or p.startswith(prefix) else prefix + p
                for i, p in enumerate(pieces)
            ]
        out.extend(pieces)
    return out


# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #
def _llama_engine():
    """按需加载 llama 引擎（延迟导入避免模块环）。"""
    from app import rag_llama

    return rag_llama


def _use_llama_engine() -> bool:
    """llama 引擎只在 Qdrant 后端下生效。

    vector_backend=db（测试/离线模式）时必须走 legacy 路径——llama 的
    QdrantVectorStore 不感知 db 后端，否则测试会把点写进真实云集合
    （实测污染事故：test_flows 把测试租户的点写进了 aics_chunks_v2）。
    """
    return settings.rag_engine == "llama" and settings.vector_backend == "qdrant"


async def index_document(
    db: Session,
    *,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    text: str,
    blocks: list[str] | None = None,
) -> IndexResult:
    """切分 + 向量化 + 写入向量库。调用方负责 commit。

    `blocks` 非空时代表解析层已经给出了结构化原子块（表格记录），原样入库；
    否则退回对 `text` 做段落滑窗。
    """
    if _use_llama_engine():
        return await _llama_engine().index_document(
            db, tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id, text=text, blocks=blocks
        )
    result = IndexResult()
    pieces = chunk_blocks(blocks) if blocks else chunk_text(text)
    if not pieces:
        raise ValueError("文档未切分出任何有效片段")

    chunk_rows: list[Chunk] = []
    for seq, piece in enumerate(pieces):
        chunk_rows.append(
            Chunk(
                id=new_id(),
                tenant_id=tenant_id,
                kb_id=kb_id,
                doc_id=doc_id,
                seq=seq,
                text=piece,
                enabled=True,
            )
        )

    # 标识符行（订单编号/运单号/SKU 等"标签: 长数字"块）不建向量：
    # 向量模型对这类块给出"万金油"高分（对任何提问都有 0.5 上下文的相似度），
    # 会把真实内容整体挤出召回池。它们只服务 _pinned_key_chunks 的精确号码
    # 查询（走 SQLite 全文定位），SQLite 照常保留，只是不进向量库。
    vector_rows = [c for c in chunk_rows if not _ID_ROW_RE.search(c.text)]
    skipped = len(chunk_rows) - len(vector_rows)
    if skipped:
        result.warnings.append(f"已跳过 {skipped} 个标识符片段的向量化（仅支持精确号码查询）")

    embed_res = await embed_texts([_strip_meta(c.text) for c in vector_rows])
    if len(embed_res.vectors) != len(vector_rows):
        raise RuntimeError("向量数量与片段数量不一致，已中止入库")
    result.tokens = embed_res.tokens

    for row in chunk_rows:
        db.add(row)
    # 先落库再写向量索引：向量库后端需要能查到这些行（否则会重复插入同一主键）。
    db.flush()

    store = build_vector_store(db, embed_res.dim or settings.embed_dim)
    store.upsert(
        [
            VectorItem(
                id=row.id,
                tenant_id=tenant_id,
                kb_id=kb_id,
                doc_id=doc_id,
                text=row.text,
                vector=vector,
            )
            for row, vector in zip(vector_rows, embed_res.vectors)
        ]
    )

    ratelimit.record_usage(
        db,
        tenant_id=tenant_id,
        employee_id=None,
        kind="embedding",
        model=settings.embed_model or settings.embed_provider,
        prompt_tokens=result.tokens,
        origin="ingest",
    )
    result.chunk_count = len(chunk_rows)
    return result


def purge_document(db: Session, tenant_id: str, doc_id: str) -> int:
    if _use_llama_engine():
        return _llama_engine().purge_document(db, tenant_id, doc_id)
    store = build_vector_store(db)
    removed = store.delete_by_doc(tenant_id, doc_id)
    rows = db.execute(
        select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.doc_id == doc_id)
    ).scalars().all()
    for row in rows:
        db.delete(row)
    return max(removed, len(rows))


def purge_kb(db: Session, tenant_id: str, kb_id: str) -> None:
    if _use_llama_engine():
        _llama_engine().purge_kb(db, tenant_id, kb_id)
        return
    store = build_vector_store(db)
    store.delete_by_kb(tenant_id, kb_id)
    for row in db.execute(
        select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.kb_id == kb_id)
    ).scalars().all():
        db.delete(row)


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def kb_ids_for_employee(db: Session, tenant_id: str, employee_id: str) -> list[str]:
    rows = db.execute(
        select(EmployeeKbBinding.kb_id).where(
            EmployeeKbBinding.tenant_id == tenant_id,
            EmployeeKbBinding.employee_id == employee_id,
        )
    ).scalars().all()
    return list(rows)


def doc_ids_for_employee(db: Session, tenant_id: str, employee_id: str) -> list[str]:
    """文件级绑定（优先于知识库绑定）。"""
    from app.models import EmployeeDocBinding

    rows = db.execute(
        select(EmployeeDocBinding.doc_id).where(
            EmployeeDocBinding.tenant_id == tenant_id,
            EmployeeDocBinding.employee_id == employee_id,
        )
    ).scalars().all()
    return list(rows)


# 提问里像是"主键"的片段：订单号/运单号这类长编号，或足够长的字母数字编码。
# 只做**原文精确包含**匹配，不做模糊语义——目的不是排序，而是保证"指名道姓"的
# 那条记录一定在上下文里（见 _pinned_key_chunks 的说明）。
_KEY_TOKEN_RE = re.compile(r"\d{8,}|[A-Za-z]{2,}[-_]?\d{6,}|[A-Za-z0-9]{14,}")


def _pinned_key_chunks(
    db: Session,
    *,
    tenant_id: str,
    query: str,
    kb_ids: list[str],
    doc_ids: list[str],
    limit: int = 3,      # 留足位置给语义命中：k 通常 5，锁 3 条足够
) -> list[RetrievedChunk]:
    """提问里出现订单号这类主键时，直接按原文锁定含它的片段。

    为什么需要：向量检索对"长串编号 + 字段名"（如「订单 5127693301594110026 的运单号」）
    表现很差——编号本身几乎没有语义，粗排会被一堆"长得像"的同字段片段挤掉
    （实测：正确那条没进 top5，捞上来的全是别的订单的运单号碎片，模型于是答"没有记录"）。
    精确匹配是确定性的：库里有就是有，命中了就强行放进上下文。
    """
    tokens = [t for t in _KEY_TOKEN_RE.findall(query or "") if len(t) >= 8]
    if not tokens:
        return []

    conds = [Chunk.tenant_id == tenant_id, Chunk.enabled.is_(True)]
    if doc_ids:
        conds.append(Chunk.doc_id.in_(doc_ids))
    elif kb_ids:
        conds.append(Chunk.kb_id.in_(kb_ids))
    else:
        return []

    seen: set[str] = set()
    found: list[tuple[Chunk, str, str]] = []
    for tok in tokens[:3]:                    # 最多查 3 个编号，避免 OR 条件过多
        rows = db.execute(
            select(Chunk, Document.filename)
            .outerjoin(Document, Document.id == Chunk.doc_id)
            .where(*conds, Chunk.text.like(f"%{tok}%"))
            .limit(60)                        # 候选池放大，下面再挑信息量最大的
        ).all()
        for chunk, filename in rows:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            found.append((chunk, filename or "", tok))

    # 同一个编号会同时出现在「完整记录块」和一堆「按列聚合的属性块」里。
    # 属性块只带该编号的一个字段，而且被切成十几片，**完整记录块才是一行全字段**，
    # 必须优先——否则锁进去的全是碎片，模型照样答"没有记录"。
    # 判别依据：编号在块内出现次数（记录块里标题和主键字段各一次）、字段分隔数。
    found.sort(
        key=lambda t: (t[0].text.count(t[2]), t[0].text.count("；")), reverse=True
    )

    out: list[RetrievedChunk] = []
    for chunk, filename, _tok in found[:limit]:
        out.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                kb_id=chunk.kb_id,
                doc_id=chunk.doc_id,
                filename=filename,
                score=1.0,                # 精确命中：压过所有语义分
                text=chunk.text,
            )
        )
    return out


def _service_supplement(
    db: Session,
    *,
    tenant_id: str,
    kb_ids: list[str],
    doc_ids: list[str],
    query: str,
    merged: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """售后服务/联系方式块常驻兜底。

    查询命中售后/维修类意图、且当前结果里没有联系方式块时，从 SQLite
    直捞一条【售后服务】块补到结果**末尾**：不挤占 top_k 名额、score 置 0，
    不参与置信度与排序，只为让模型在建议"联系售后"时有号码可写。
    """
    if any(_SERVICE_BLOCK_RE.match(c.text) for c in merged):
        return merged
    if not _SERVICE_QUERY_RE.search(query):
        return merged
    stmt = (
        select(Chunk, Document.filename)
        .outerjoin(Document, Document.id == Chunk.doc_id)
        .where(
            Chunk.tenant_id == tenant_id,
            Chunk.enabled.is_(True),
            Chunk.text.like("【售后服务】%"),
        )
    )
    if doc_ids:
        stmt = stmt.where(Chunk.doc_id.in_(doc_ids))
    elif kb_ids:
        stmt = stmt.where(Chunk.kb_id.in_(kb_ids))
    else:
        return merged
    row = db.execute(stmt.limit(1)).first()
    if row is None:
        return merged
    chunk, filename = row
    logger.info("售后服务块兜底注入 chunk=%s", chunk.id)
    merged.append(
        RetrievedChunk(
            chunk_id=chunk.id,
            kb_id=chunk.kb_id,
            doc_id=chunk.doc_id,
            filename=filename or "",
            score=0.0,
            text=chunk.text,
        )
    )
    return merged


async def retrieve(
    db: Session,
    *,
    tenant_id: str,
    kb_ids: list[str],
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    doc_ids: list[str] | None = None,
    no_threshold: bool = False,
) -> tuple[list[RetrievedChunk], int]:
    """返回 (命中片段, 向量调用消耗 token)。

    no_threshold=True：0 命中兜底的二次检索用——目的不是"过滤"而是"捞回与提问
    最相邻的几条"，所以跳过二阶段重排（省一次模型调用，也避免精排把低分块全砍光），
    并忽略所有分数门槛。

    两种范围模式：
    - **文件级**（doc_ids 非空）：用户逐个勾选了文件，直接按 doc_id 过滤全局排序取 top_k；
    - **库级**（kb_ids）：按库独立检索——每库各取保底候选（settings.per_kb_top_k，
      单库时取满 top_k），再合并按分数排序，避免大库把小库高分段挤出结果。
    向量只算一次（embed_query），多库/多文件只是多几次向量库查询，不增加模型调用量。
    """
    if _use_llama_engine():
        return await _llama_engine().retrieve(
            db, tenant_id=tenant_id, kb_ids=kb_ids, query=query,
            top_k=top_k, min_score=min_score, doc_ids=doc_ids,
            no_threshold=no_threshold,
        )
    if not tenant_id:
        raise ValueError("retrieve 必须携带 tenant_id")
    doc_ids = [d for d in (doc_ids or []) if d]
    if not doc_ids and not kb_ids:
        return [], 0

    # 精确主键命中（订单号/运单号等）：不参与向量排序，直接锁定
    pinned = _pinned_key_chunks(
        db, tenant_id=tenant_id, query=query, kb_ids=kb_ids, doc_ids=doc_ids
    )
    pinned_ids = {c.chunk_id for c in pinned}

    vec, tokens = await embed_query(query)
    store = build_vector_store(db, len(vec))
    k = top_k or settings.top_k
    ms = min_score if min_score is not None else settings.min_score

    # 启用重排时走二阶段：粗排放宽（候选池要大），精排再收紧。
    # 粗排拿 rerank_candidates 条、门槛用 rerank_recall_min_score；
    # 精排后按 rerank_min_score 过滤、取最终 k 条。
    # no_threshold 模式下跳过重排：见函数 docstring。
    two_stage = rerank_enabled() and not no_threshold
    recall_k = max(k, settings.rerank_candidates) if two_stage else k
    recall_ms = min(ms, settings.rerank_recall_min_score) if two_stage else ms
    final_ms = settings.rerank_min_score if two_stage else ms

    if doc_ids:
        hits = store.search(tenant_id, vec, top_k=recall_k * 3, doc_ids=doc_ids, min_score=recall_ms)
    elif len(kb_ids) <= 1:
        hits = store.search(tenant_id, vec, top_k=recall_k * 3, kb_ids=kb_ids, min_score=recall_ms)
    else:
        # 多库：每库独立取保底候选（保底 = max(per_kb_top_k, top_k/库数)），
        # 合并后交给下游 dedupe_hits 统一去重并限总量。
        per_kb = max(settings.per_kb_top_k, math.ceil(recall_k / len(kb_ids)))
        hits = []
        for kb_id in kb_ids:
            hits.extend(
                store.search(tenant_id, vec, top_k=per_kb * 3, kb_ids=[kb_id], min_score=recall_ms)
            )
        hits.sort(key=lambda h: h.score, reverse=True)
    if not hits:
        merged = _service_supplement(
            db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
            query=query, merged=list(pinned),
        )
        return merged, tokens

    ids = [h.chunk_id for h in hits]
    rows = db.execute(
        select(Chunk, Document.filename)
        .outerjoin(Document, Document.id == Chunk.doc_id)
        .where(
            Chunk.tenant_id == tenant_id,          # 双保险：再次限定租户
            Chunk.enabled.is_(True),
            Chunk.id.in_(ids),
        )
    ).all()
    by_id = {chunk.id: (chunk, filename or "") for chunk, filename in rows}

    out: list[RetrievedChunk] = []
    for hit in hits:
        found = by_id.get(hit.chunk_id)
        if found is None:
            # 命中了不属于本租户（或已被删除）的片段，直接丢弃
            logger.warning("检索命中越权片段已丢弃 chunk=%s tenant=%s", hit.chunk_id, tenant_id)
            continue
        chunk, filename = found
        if chunk.id in pinned_ids:
            continue      # 已由精确匹配锁定，避免重复占位
        out.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                kb_id=chunk.kb_id,
                doc_id=chunk.doc_id,
                filename=filename,
                score=hit.score,
                text=chunk.text,
            )
        )
    if not out:
        return pinned[:k], tokens

    # 标识符行（订单编号/运单号/SKU 等"标签: 长数字"行）只为精确号码查询服务。
    # 查询本身不带长数字串时，它们是语义检索的"万金油"噪声——对任何提问都有
    # 0.5 上下文的向量分，会把真实内容整体挤出候选池（实测存放/休眠类问题
    # 全军覆没）。号码查询场景由 _pinned_key_chunks 的精确命中负责。
    if not _QUERY_HAS_LONG_ID.search(query):
        out = [c for c in out if not _ID_ROW_RE.search(c.text)]
    # 粗取放大过 3 倍，剔除噪声后按向量分收回候选池预算，rerank 条数不变。
    # 只在溢出时才排序裁剪——常规情况下保持原顺序，不扰动 rerank 输入。
    if len(out) > recall_k:
        out.sort(key=lambda c: c.score, reverse=True)
        out = out[:recall_k]

    if two_stage:
        scores = await rerank_passages(query, [_strip_meta(c.text) for c in out])
        if scores is not None:
            ratelimit.record_usage(
                db,
                tenant_id=tenant_id,
                employee_id=None,
                kind="rerank",
                model=settings.rerank_model,
                prompt_tokens=sum(len(c.text) for c in out) // 4,   # 粗略折算，便于看趋势
                origin="retrieve",
            )
            ordered = sorted(zip(scores, range(len(out))), key=lambda x: (-x[0], x[1]))
            reranked: list[RetrievedChunk] = []
            for sc, idx in ordered:
                if sc < final_ms:
                    # 精排判定为离题：宁可少给一条，也不让噪声进提示词
                    continue
                c = out[idx]
                c.score = sc          # 覆盖为精排分，下游排序与展示都以它为准
                reranked.append(c)
                if len(reranked) >= k:
                    break
            return _service_supplement(
                db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
                query=query, merged=_merge_pinned(pinned, reranked, k),
            ), tokens
        # 重排不可用：粗排是**放宽过**的（门槛更低、条数更多），
        # 必须按原来的 min_score 收紧，否则降级反而会灌进更多噪声。
        logger.info("重排降级：按原阈值 %.2f 收紧候选（粗排门槛 %.2f）", ms, recall_ms)
        return _service_supplement(
            db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
            query=query, merged=_merge_pinned(pinned, [c for c in out if c.score >= ms], k),
        ), tokens

    return _service_supplement(
        db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
        query=query, merged=_merge_pinned(pinned, out, k),
    ), tokens


def _merge_pinned(
    pinned: list[RetrievedChunk], rest: list[RetrievedChunk], k: int
) -> list[RetrievedChunk]:
    """精确命中的片段排最前，剩下的按分数补足到 k 条。"""
    if not pinned:
        return rest[:k]
    merged = list(pinned)
    for c in rest:
        if len(merged) >= k:
            break
        merged.append(c)
    return merged[:k]


def dedupe_hits(hits: list[RetrievedChunk], limit: int | None = None) -> list[RetrievedChunk]:
    """按文本去重并按分数排序，控制注入条数（PRD 3.2：建议 5 条）。"""
    limit = limit or settings.top_k
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for hit in sorted(hits, key=lambda h: h.score, reverse=True):
        key = hit.text.strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out
