"""RAG 标准化引擎：基于 LlamaIndex 的检索实现（RAG_ENGINE=llama）。

迁移路线 A 的落点：检索编排交给 LlamaIndex 标准组件——
- ``QdrantVectorStore``（官方集成，flat_metadata=True 让我们的隔离字段落在顶层 payload）
- ``BM25Retriever``（bm25s 内核，jieba 预分词的中文适配）
- ``QueryFusionRetriever``（双路召回 + 标准 RRF 倒数排名融合，替代旧引擎的"纯向量 + 3 倍粗取"）
- ``BaseRetriever`` 自定义子类承载向量路与 BM25 路（标准扩展点）

**刻意保留的自研层**（它们是踩坑成果，不属于编排层）：
- 切分：``chunk_text`` / ``chunk_blocks`` / 要点清单等 textparse 全套——作为入库前置，产 TextNode
- 精排：Qwen3-Reranker 交叉编码（rerank.py）在融合结果上做第二阶段
- 主键锁定 / 售后兜底 / 置信度语义：SQLite 逻辑，留在 rag.py 的包装层

与旧引擎的已知差异：
- 多库检索的"每库保底名额"未实现（融合检索天然按全局排名）；单库场景等价
- 粗排分数门槛改为向量路内部过滤（RRF 融合分与余弦不同量纲，不能出融合后再滤）
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any

import jieba
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import ratelimit
from app.config import settings
from app.embedding import embed_query, embed_texts
from app.models import Chunk, Document
from app.resilience import Breaker
from app.vectorstore.qdrant_store import VectorStoreUnavailable, _is_transient
import time
from app.rag import (
    _ID_ROW_RE,
    _QUERY_HAS_LONG_ID,
    _merge_pinned,
    _pinned_key_chunks,
    _service_supplement,
    _strip_meta,
    IndexResult,
    RetrievedChunk,
    chunk_blocks,
    chunk_text,
)
from app.rerank import RerankUnavailable
from app.rerank import enabled as rerank_enabled
from app.rerank import rerank_scores

jieba.setLogLevel(60)  # 关闭 jieba 首次加载的建库日志

# 本引擎只用 LlamaIndex 的检索编排组件，LLM 一律走 app/llm.py 自己的双档位适配。
# 不显式置 None 的话，QueryFusionRetriever 构造会惰性解析"默认 LLM"并要求安装
# llama-index-llms-openai（ImportError）。置空 = 声明我们不需要它的 LLM。
from llama_index.core import Settings as _LISettings

_LISettings.llm = None

logger = logging.getLogger("aics.rag_llama")


# --------------------------------------------------------------------------- #
# 向量库（llama QdrantVectorStore，集合与旧引擎隔离）
# --------------------------------------------------------------------------- #
_LLAMA_STORE: QdrantVectorStore | None = None
_LLAMA_CLIENT: Any = None
_STORE_LOCK = threading.Lock()


def llama_vector_store() -> QdrantVectorStore:
    """单例 llama QdrantVectorStore（连接复用 + 集合/payload 索引就绪）。"""
    global _LLAMA_STORE, _LLAMA_CLIENT
    with _STORE_LOCK:
        if _LLAMA_STORE is not None:
            return _LLAMA_STORE
        from qdrant_client import QdrantClient

        if not settings.qdrant_url:
            raise RuntimeError("RAG_ENGINE=llama 需要配置 QDRANT_URL")
        _LLAMA_CLIENT = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
            prefer_grpc=False,
        )
        _LLAMA_STORE = QdrantVectorStore(
            client=_LLAMA_CLIENT,
            collection_name=settings.qdrant_collection_llama,
            flat_metadata=True,  # metadata 平铺到顶层 payload，过滤键直接可索引
        )
        _ensure_collection(_LLAMA_CLIENT, settings.qdrant_collection_llama)
        _ensure_payload_indexes(_LLAMA_CLIENT, settings.qdrant_collection_llama)
        return _LLAMA_STORE


def _ensure_collection(client: Any, collection: str) -> None:
    """集合不存在就按 EMBED_DIM 显式创建（llama 只在首次 add 时隐式建，
    会导致删除/建索引先于建集合发生而 404—— purge 必须能在空集合上工作）。"""
    from qdrant_client import models

    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=settings.embed_dim, distance=models.Distance.COSINE
        ),
    )
    logger.info("已创建 llama 引擎集合 %s（dim=%s）", collection, settings.embed_dim)


def _ensure_payload_indexes(client: Any, collection: str) -> None:
    """隔离字段建 keyword 索引（幂等）。缺失时 Qdrant 过滤直接 400，不是慢查询。"""
    from qdrant_client import models

    for field in ("tenant_id", "kb_id", "doc_id"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            if "already exists" not in str(exc):
                logger.warning("创建 payload 索引 %s 未成功：%s", field, exc)


def _purge_llama_points(tenant_id: str, extra: dict[str, str]) -> None:
    """按 payload 条件删 llama 集合的点（flat_metadata 字段在顶层）。

    两条「必须吞掉异常」的理由，都是为了让删除接口不要 500：

    1. 集合还不存在 = 必然没有点，直接跳过（否则 delete 404 会让文档删除整条 500）。
    2. **Qdrant 不可达也不能拦下删除**。本实例在公网（Qdrant Cloud），
       一次 TLS 抖动（UNEXPECTED_EOF_WHILE_READING）就会让 `llama_vector_store()`
       构造或 delete 直接抛异常 —— 而删除走的是「先 purge 向量、再 commit SQLite」，
       异常一路冒到路由层就是 500，**SQLite 事务不提交，文档原封不动留在列表里**，
       用户看到的是「删除失败」且反复重试永远失败。
       向量清理失败只留下孤儿点，而检索侧本来就有 SQLite 双保险
       （见 retrieve：命中 chunk_id 必须在 chunks 表里查得到，否则丢弃），
       孤儿点永远不会被召回。所以「删不掉的向量」远比「删不掉的文档」无害 ——
       这里必须降级为告警，把删除本身让出去。
    """
    from qdrant_client import models

    try:
        store = llama_vector_store()
        if not store._client.collection_exists(store.collection_name):
            return
        must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
        for key, value in extra.items():
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        _QdrantFlaky.call(
            "delete_by_filter",
            store._client.delete,
            collection_name=store.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001 - 向量清理失败不能连累文档删除
        logger.warning(
            "llama 集合向量清理失败（已跳过，SQLite 侧照常删除）filter=%s：%s", extra, exc
        )


# --------------------------------------------------------------------------- #
# 节点构建（TextNode 是 llama 的标准数据单元）
# --------------------------------------------------------------------------- #
def _node_id(chunk_id: str) -> str:
    """chunk 32 位 hex → UUID 字符串。Qdrant point id 只收 UUID/整数，
    llama 集成不做转换，所以必须在建节点时自己转，检索时再转回来。"""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def _chunk_id_from_node(node_id: str) -> str:
    try:
        return uuid.UUID(node_id).hex
    except (ValueError, AttributeError, TypeError):
        return node_id


def _make_node(chunk_id: str, text: str, *, tenant_id: str, kb_id: str, doc_id: str, seq: int, vector: list[float] | None = None) -> TextNode:
    node = TextNode(
        id_=_node_id(chunk_id),
        text=text,
        metadata={
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "seq": seq,
        },
        embedding=vector,
    )
    # ref_doc_id 是 relationships[SOURCE] 的只读视图，构造器不直接收参；
    # 它决定 node_to_metadata_dict 的 document_id，是按文档删除点的依据。
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


# --------------------------------------------------------------------------- #
# 入库
# --------------------------------------------------------------------------- #
async def index_document(
    db: Session,
    *,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    text: str,
    blocks: list[str] | None = None,
) -> IndexResult:
    """切分 + 向量化 + 写 SQLite 与 llama Qdrant 集合。调用方负责 commit。

    切分逻辑与旧引擎完全一致（textparse 的结构化块 / 段落滑窗）；
    差异只在向量的落点：llama 集合、TextNode 载荷（标准 schema）。
    """
    result = IndexResult()
    pieces = chunk_blocks(blocks) if blocks else chunk_text(text)
    if not pieces:
        raise ValueError("文档未切分出任何有效片段")

    chunk_rows: list[Chunk] = []
    for seq, piece in enumerate(pieces):
        chunk_rows.append(
            Chunk(
                id=_new_chunk_id(),
                tenant_id=tenant_id,
                kb_id=kb_id,
                doc_id=doc_id,
                seq=seq,
                text=piece,
                enabled=True,
            )
        )

    # 标识符行不建向量——与旧引擎同一决策（万金油噪声挤占召回池）
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
    db.flush()

    nodes = [
        _make_node(
            row.id,
            row.text,
            tenant_id=tenant_id,
            kb_id=kb_id,
            doc_id=doc_id,
            seq=row.seq,
            vector=vec,
        )
        for row, vec in zip(vector_rows, embed_res.vectors)
    ]
    if nodes:
        llama_vector_store().add(nodes)

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


def _new_chunk_id() -> str:
    from app.utils import new_id

    return new_id()


def purge_document(db: Session, tenant_id: str, doc_id: str) -> int:
    _purge_llama_points(tenant_id, {"doc_id": doc_id})
    rows = db.execute(
        select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.doc_id == doc_id)
    ).scalars().all()
    for row in rows:
        db.delete(row)
    return len(rows)


def purge_kb(db: Session, tenant_id: str, kb_id: str) -> None:
    _purge_llama_points(tenant_id, {"kb_id": kb_id})
    for row in db.execute(
        select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.kb_id == kb_id)
    ).scalars().all():
        db.delete(row)


# --------------------------------------------------------------------------- #
# 检索器（标准扩展点：BaseRetriever 子类）
# --------------------------------------------------------------------------- #
def _scope_filters(tenant_id: str, kb_ids: list[str], doc_ids: list[str]) -> MetadataFilters:
    """租户隔离 + 库/文件范围 → llama MetadataFilters（flat payload 顶层键）。"""
    conds = [
        MetadataFilter(key="tenant_id", value=tenant_id, operator=FilterOperator.EQ)
    ]
    if doc_ids:
        conds.append(MetadataFilter(key="doc_id", value=doc_ids, operator=FilterOperator.IN))
    elif kb_ids:
        conds.append(MetadataFilter(key="kb_id", value=kb_ids, operator=FilterOperator.IN))
    return MetadataFilters(filters=conds)


class _QdrantFlaky:
    """llama 引擎的 Qdrant 调用韧性层：退避重试 + 熔断。

    为什么必须有：Qdrant Cloud 在公网上，一次 TLS 重置就会让整个向量路
    抛异常 → 检索 0 命中 → 客户收到"没查到资料"。legacy 引擎在
    vectorstore/qdrant_store._call 里有同样的保护，移植时不能丢。
    """

    _breaker: Breaker | None = None

    @classmethod
    def _bk(cls) -> Breaker:
        if cls._breaker is None:
            cls._breaker = Breaker(
                threshold=settings.qdrant_breaker_threshold,
                cooldown=settings.qdrant_breaker_cooldown,
            )
        return cls._breaker

    @classmethod
    def call(cls, op: str, fn, *args, **kwargs):
        breaker = cls._bk()
        if breaker.is_open():
            raise VectorStoreUnavailable(
                "向量检索服务暂时不可用，请稍后重试",
                detail="circuit breaker open（llama 引擎）",
            )
        attempts = max(1, settings.qdrant_retries)
        delay = max(0.0, settings.qdrant_retry_delay)
        last: BaseException | None = None
        for i in range(attempts):
            try:
                result = fn(*args, **kwargs)
                breaker.on_success()
                return result
            except Exception as exc:
                last = exc
                if not _is_transient(exc) or i == attempts - 1:
                    break
                logger.warning(
                    "Qdrant(llama) %s 第 %d/%d 次失败（将重试）：%s", op, i + 1, attempts, exc
                )
                time.sleep(delay * (2 ** i))
        if _is_transient(last):
            breaker.on_failure()
        raise last


class AicsVectorRetriever(BaseRetriever):
    """向量路检索器：llama QdrantVectorStore 标准查询接口。

    embedding 由调用方预算好塞进 QueryBundle（避免在同步 _retrieve 里调异步 API）；
    cosine 分数在路内先按粗排门槛过滤——RRF 融合分与余弦不同量纲，出融合再滤就晚了。
    """

    _store: Any = None
    _filters: Any = None
    _top_k: int = 0
    _min_score: float = 0.0

    def __init__(self, store: Any, filters: MetadataFilters, top_k: int, min_score: float, **kw: Any):
        super().__init__(**kw)
        self._store = store
        self._filters = filters
        self._top_k = top_k
        self._min_score = min_score

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vec = query_bundle.embedding
        if vec is None:
            vec = _sync_embed_one(query_bundle.query_str)
        res = _QdrantFlaky.call(
            "query",
            self._store.query,
            VectorStoreQuery(
                query_embedding=vec,
                similarity_top_k=self._top_k,
                filters=self._filters,
            ),
        )
        out: list[NodeWithScore] = []
        for node, score in zip(res.nodes or [], res.similarities or []):
            if self._min_score > 0 and float(score) < self._min_score:
                continue
            out.append(NodeWithScore(node=node, score=float(score)))
        return out


class JiebaBM25Retriever(BM25Retriever):
    """BM25 路检索器：bm25s 内核 + jieba 分词的中文适配。

    llama 官方 BM25Retriever 的默认分词对中文无效（\\w\\w+ 会把整句当一个 token）。
    这里节点文本与查询都过 jieba 后再按空格切开（token_pattern 放宽到 \\w+），
    两侧分词一致，BM25 统计才有意义。
    """

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        spaced = QueryBundle(
            query_str=" ".join(jieba.lcut(query_bundle.query_str or ""))
        )
        return super()._retrieve(spaced)


_BM25_CACHE: dict[tuple, Any] = {}
_BM25_LOCK = threading.Lock()


def _scope_chunks(
    db: Session, *, tenant_id: str, kb_ids: list[str], doc_ids: list[str]
) -> list[Chunk]:
    conds = [Chunk.tenant_id == tenant_id, Chunk.enabled.is_(True)]
    if doc_ids:
        conds.append(Chunk.doc_id.in_(doc_ids))
    elif kb_ids:
        conds.append(Chunk.kb_id.in_(kb_ids))
    else:
        return []
    return db.execute(select(Chunk).where(*conds)).scalars().all()


def _bm25_retriever(
    db: Session,
    *,
    tenant_id: str,
    kb_ids: list[str],
    doc_ids: list[str],
    top_k: int,
) -> BaseRetriever | None:
    """BM25 路：语料来自 SQLite（唯一真源），按范围缓存，块数变化即重建。"""
    rows = _scope_chunks(db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids)
    rows = [c for c in rows if not _ID_ROW_RE.search(c.text)]
    if not rows:
        return None
    cache_key = (tenant_id, tuple(sorted(doc_ids or kb_ids)), len(rows))
    with _BM25_LOCK:
        hit = _BM25_CACHE.get(cache_key)
    if hit is not None:
        hit[0]._similarity_top_k = top_k
        return hit[0]

    nodes: list[BaseNode] = []
    for c in rows:
        nodes.append(
            TextNode(
                id_=_node_id(c.id),
                text=" ".join(jieba.lcut(_strip_meta(c.text))),
            )
        )
    retriever = JiebaBM25Retriever(
        nodes=nodes,
        similarity_top_k=top_k,
        token_pattern=r"\w+",
        skip_stemming=True,
        language="en",
        verbose=False,
    )
    with _BM25_LOCK:
        _BM25_CACHE.clear()          # 作用域数量有限，直接整表换新防膨胀
        _BM25_CACHE[cache_key] = (retriever, len(rows))
    return retriever


# --------------------------------------------------------------------------- #
# 同步向量兜底（QueryBundle.embedding 缺失时 / 单测）
# --------------------------------------------------------------------------- #
def _sync_embed_one(text: str) -> list[float]:
    if settings.embed_provider == "local":
        from app.embedding import _local_embed_one

        return _local_embed_one(text, settings.embed_dim)
    import httpx

    with httpx.Client(
        timeout=settings.embed_timeout,
        headers={"Authorization": f"Bearer {settings.embed_api_key}"},
    ) as client:
        resp = client.post(
            f"{settings.embed_base_url.rstrip('/')}/embeddings",
            json={"model": settings.embed_model, "input": [text or " "]},
        )
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
        return list(map(float, items[0]["embedding"]))


# --------------------------------------------------------------------------- #
# 检索主入口（对齐旧引擎 rag.retrieve 的契约）
# --------------------------------------------------------------------------- #
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
    if not tenant_id:
        raise ValueError("retrieve 必须携带 tenant_id")
    doc_ids = [d for d in (doc_ids or []) if d]
    if not doc_ids and not kb_ids:
        return [], 0

    # 主键精确锁定（SQLite 通道，与旧引擎一致）
    pinned = _pinned_key_chunks(
        db, tenant_id=tenant_id, query=query, kb_ids=kb_ids, doc_ids=doc_ids
    )
    pinned_ids = {c.chunk_id for c in pinned}

    vec, tokens = await embed_query(query)
    qb = QueryBundle(query_str=query, embedding=vec)

    k = top_k or settings.top_k
    ms = min_score if min_score is not None else settings.min_score
    two_stage = rerank_enabled() and not no_threshold
    recall_k = max(k, settings.rerank_candidates) if two_stage else k
    recall_ms = min(ms, settings.rerank_recall_min_score) if two_stage else ms
    final_ms = settings.rerank_min_score if two_stage else ms

    # 向量路可用性探测。
    # 为什么必须单独探：llama_vector_store() 的**构造阶段**就联网（QdrantVectorStore
    # init 内部做集合探测与 index 升级），而熔断器只包住了 _QdrantFlaky.call，覆盖不到
    # 构造期。Qdrant Cloud 走公网（曾经配在 sa-east-1，从国内被主动 RST 阻断），这里
    # 一抛，整条检索连同 BM25 路一起陪葬，客户收到"没查到资料"——但那份资料明明在
    # SQLite 里。BM25 路的语料来自 SQLite（本地唯一真源），向量路挂了它必须能独立顶上。
    vec_ret: AicsVectorRetriever | None = None
    try:
        store = llama_vector_store()
        vec_ret = AicsVectorRetriever(
            store=store, filters=_scope_filters(tenant_id, kb_ids, doc_ids),
            top_k=recall_k * 3, min_score=recall_ms,
        )
    except Exception as exc:  # noqa: BLE001 - 向量路不可用不能连累 BM25 路
        logger.warning(
            "llama 向量库不可用，本轮降级为纯 BM25 检索（SQLite 本地语料，不受影响）：%s", exc
        )

    bm = _bm25_retriever(
        db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids, top_k=recall_k
    )
    retrievers: list[BaseRetriever] = [r for r in (vec_ret, bm) if r is not None]
    if not retrievers:
        raise VectorStoreUnavailable(
            "向量检索服务暂时不可用，请稍后重试",
            detail="向量路与 BM25 路均不可用（llama 引擎）",
        )

    if len(retrievers) == 1:
        # 单路无需融合：少一层黑盒，也避免融合器「全有或全无」把唯一能跑的那路放大成失败
        fused = await retrievers[0].aretrieve(qb)
    else:
        fusion = QueryFusionRetriever(
            retrievers,
            similarity_top_k=recall_k,
            num_queries=1,          # 查询改写属 P3，标准件已就位（>1 即启用）
            mode=FUSION_MODES.RECIPROCAL_RANK,  # 0.14 默认是 SIMPLE，必须显式指定 RRF
            use_async=True,
            verbose=False,
        )
        try:
            fused = await fusion.aretrieve(qb)
        except Exception as exc:  # noqa: BLE001 - 融合器任一路抛异常就炸全链，必须单路兜底
            logger.warning("RRF 融合异常（%s），改用单路检索结果", exc)
            fused = []
            for retriever in retrievers:
                try:
                    fused = await retriever.aretrieve(qb)
                    break
                except Exception:  # noqa: BLE001 - 换下一路试
                    logger.warning("单路检索失败，跳过：%s", type(retriever).__name__)

    # 融合结果 → SQLite 双保险校验 → RetrievedChunk
    ids_hex = [_chunk_id_from_node(n.node.node_id) for n in fused]
    rows = db.execute(
        select(Chunk, Document.filename)
        .outerjoin(Document, Document.id == Chunk.doc_id)
        .where(
            Chunk.tenant_id == tenant_id,
            Chunk.enabled.is_(True),
            Chunk.id.in_(ids_hex),
        )
    ).all()
    by_id = {chunk.id: (chunk, filename or "") for chunk, filename in rows}

    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for n in fused:
        cid = _chunk_id_from_node(n.node.node_id)
        if cid in pinned_ids or cid in seen:
            continue
        found = by_id.get(cid)
        if found is None:
            logger.warning("检索命中越权片段已丢弃 chunk=%s tenant=%s", cid, tenant_id)
            continue
        seen.add(cid)  # RRF 按 node.hash 去重，向量/BM25 两路节点 hash 不同，这里兜底
        chunk, filename = found
        out.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                kb_id=chunk.kb_id,
                doc_id=chunk.doc_id,
                filename=filename,
                score=float(n.score or 0.0),
                text=chunk.text,
            )
        )
    if not out:
        merged = _service_supplement(
            db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
            query=query, merged=list(pinned),
        )
        return merged, tokens

    # 标识符噪声过滤（查询带长编号时放行，与旧引擎一致）
    if not _QUERY_HAS_LONG_ID.search(query):
        out = [c for c in out if not _ID_ROW_RE.search(c.text)]
    if len(out) > recall_k:
        out = out[:recall_k]

    if two_stage:
        try:
            scores = await rerank_scores(query, [_strip_meta(c.text) for c in out])
        except RerankUnavailable as exc:
            logger.warning("重排降级（%s）：回退向量粗排分（阈值 %.2f）", exc, ms)
            scores = None
        except Exception:  # noqa: BLE001 - 重排失败不允许炸穿整条检索链路
            logger.warning("重排异常降级：回退向量粗排分（阈值 %.2f）", ms, exc_info=True)
            scores = None
        if scores is not None:
            ratelimit.record_usage(
                db,
                tenant_id=tenant_id,
                employee_id=None,
                kind="rerank",
                model=settings.rerank_model,
                prompt_tokens=sum(len(c.text) for c in out) // 4,
                origin="retrieve",
            )
            ordered = sorted(zip(scores, range(len(out))), key=lambda x: (-x[0], x[1]))
            reranked: list[RetrievedChunk] = []
            for sc, idx in ordered:
                if sc < final_ms:
                    continue
                c = out[idx]
                c.score = sc
                reranked.append(c)
                if len(reranked) >= k:
                    break
            return _service_supplement(
                db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
                query=query, merged=_merge_pinned(pinned, reranked, k),
            ), tokens
        # 重排不可用：融合分是 RRF 排名量纲（≈1/(60+rank)），与置信度阈值量纲
        # 不通，不能像 legacy 那样拿融合分按 ms 收紧——必须回取向量路的原始
        # 相似度分（复用同一 QueryBundle，embedding 不重算，只多一次 Qdrant 查询）。
        # 向量路本身不可用时没有余弦分可回取：此时 out 已是 BM25 结果（bm25s 分数量纲，
        # 与余弦阈值 ms 同样不通），只能原样放行，不能套用向量阈值误杀。
        if vec_ret is None:
            logger.warning(
                "重排与向量路均不可用：放行 %d 条 BM25 候选（不套用余弦阈值）", len(out)
            )
            return _service_supplement(
                db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
                query=query, merged=_merge_pinned(pinned, out, k),
            ), tokens
        coarse = await vec_ret.aretrieve(qb)
        fb: list[RetrievedChunk] = []
        fb_seen: set[str] = set()
        noise_filter_on = not _QUERY_HAS_LONG_ID.search(query)
        for n in coarse:
            cid = _chunk_id_from_node(n.node.node_id)
            if cid in pinned_ids or cid in fb_seen:
                continue
            found = by_id.get(cid)
            if found is None:
                continue
            chunk, filename = found
            if noise_filter_on and _ID_ROW_RE.search(chunk.text):
                continue
            sc = float(n.score or 0.0)
            if sc < ms:  # 粗排是放宽过的（recall_ms），按原阈值收紧
                continue
            fb_seen.add(cid)
            fb.append(
                RetrievedChunk(
                    chunk_id=chunk.id, kb_id=chunk.kb_id, doc_id=chunk.doc_id,
                    filename=filename, score=sc, text=chunk.text,
                )
            )
        fb.sort(key=lambda c: -c.score)
        fb = fb[:k]
        logger.info("重排降级完成：向量粗排回收 %d 条候选", len(fb))
        return _service_supplement(
            db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
            query=query, merged=_merge_pinned(pinned, fb, k),
        ), tokens

    return _service_supplement(
        db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
        query=query, merged=_merge_pinned(pinned, out, k),
    ), tokens
