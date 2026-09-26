"""VECTOR_BACKEND=qdrant：生产推荐。用 payload 过滤做租户隔离。

需要：pip install qdrant-client

两个工程细节（都踩过）：
1. **客户端复用**：`build_vector_store` 在 `rag.retrieve` 里是**逐请求**调用的，
   若每次 new 一个 QdrantClient 并 get_collections()，等于每条消息都往 Qdrant Cloud
   多打一次元数据请求。这里用模块级缓存复用连接，并把「集合已就绪」的校验结果也缓存住。
2. **维度校验**：Qdrant 集合一旦建好维度就固定。若换了 embedding 模型（如 512 → 1536）
   而集合没重建，检索会报错或静默返回垃圾。这里在构造期就炸，并给出可执行的修法。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid

from app.config import settings
from app.resilience import Breaker
from app.resilience import RETRYABLE_STATUS as _RETRYABLE_STATUS
from app.vectorstore.base import (
    VectorHit,
    VectorItem,
    VectorStore,
    VectorStoreUnavailable,
    l2_normalize,
)

logger = logging.getLogger("aics.vectorstore.qdrant")

# url/api_key -> 客户端（连接池复用）
_CLIENTS: dict[tuple[str, str], object] = {}
# (url, collection) -> 已确认就绪的维度
_READY: dict[tuple[str, str], int] = {}
_LOCK = threading.Lock()

# 熔断器：与向量模型共用同一套实现（app/resilience.py）
_Breaker = Breaker


def _is_transient(exc: BaseException) -> bool:
    """区分「网络抖动/上游 5xx」与「请求本身有问题」。

    关键：4xx（除 429）代表请求写错了，重试一万次也没用，必须立刻抛出，
    否则会把一个明显的代码 bug 伪装成「向量库不稳定」，排查时会被带偏。
    """
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    if isinstance(exc, ResponseHandlingException):
        # 底层是 httpx/ssl 的错误：TLS 重置、连接被断、超时
        return True
    if isinstance(exc, UnexpectedResponse):
        code = getattr(exc, "status_code", None)
        return code in _RETRYABLE_STATUS
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _point_id(chunk_id: str) -> str:
    """Qdrant 的 point id 必须是 UUID 或整数；chunk_id 是 32 位 hex，转 UUID。"""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


class DimensionMismatch(VectorStoreUnavailable):
    """集合维度与当前 embedding 维度不一致——必须重建集合，否则检索结果无意义。

    继承 VectorStoreUnavailable 是为了让上层统一按「向量层不可用」处理（503 / 降级），
    但 message 要给到可执行的修法，而不是一句「稍后重试」。
    """


# 为什么需要熔断：向量库真的挂了（不是抖一下）时，如果每条客户消息都要走完
# qdrant_retries 次退避重试才降级，客户会**既等得久又拿不到答案**——两头都亏。
# 熔断窗口内立刻降级（快速给出转人工话术），冷却后再放一个探针请求过去。

# (url, collection) -> 熔断器
_BREAKERS: dict[tuple[str, str], _Breaker] = {}


def _breaker_for(key: tuple[str, str]) -> _Breaker:
    with _LOCK:
        b = _BREAKERS.get(key)
        if b is None:
            b = _Breaker(
                threshold=settings.qdrant_breaker_threshold,
                cooldown=settings.qdrant_breaker_cooldown,
            )
            _BREAKERS[key] = b
        return b


class QdrantVectorStore(VectorStore):
    name = "qdrant"

    def __init__(self, dim: int):
        try:
            from qdrant_client import QdrantClient  # 延迟导入
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "VECTOR_BACKEND=qdrant 需要安装 qdrant-client：pip install qdrant-client"
            ) from exc

        if not settings.qdrant_url:
            raise RuntimeError("VECTOR_BACKEND=qdrant 需要配置 QDRANT_URL")
        if not dim or dim <= 0:
            raise RuntimeError(f"向量维度非法：{dim}（检查 EMBED_DIM）")

        ck = (settings.qdrant_url, settings.qdrant_api_key)
        with _LOCK:
            client = _CLIENTS.get(ck)
            if client is None:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=getattr(settings, "qdrant_timeout", 20.0),
                    prefer_grpc=False,
                )
                _CLIENTS[ck] = client
        self._client = client
        self._collection = settings.qdrant_collection
        self._dim = dim
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    def _call(self, op: str, fn, *args, **kwargs):
        """带熔断 + 退避重试地调用 Qdrant。

        为什么必须重试：这个实例跑在公网上（Qdrant Cloud），一次 TLS 重置
        （`UNEXPECTED_EOF_WHILE_READING`）就会让 1 条客户消息降级成「答不上来」。
        对抖动做重试，成本极低，收益是客户侧完全无感。

        为什么要熔断：如果是真故障（不是抖一下），逐请求重试会让每条消息都白等
        1.5 秒还照样答不上来。熔断后立刻降级、冷却后再探。
        """
        breaker = _breaker_for((settings.qdrant_url, self._collection))
        if breaker.is_open():
            raise VectorStoreUnavailable(
                "向量检索服务暂时不可用，请稍后重试",
                detail="circuit breaker open（连续失败后进入冷却，已跳过网络请求）",
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
                    "Qdrant %s 第 %d/%d 次失败（将重试）：%s", op, i + 1, attempts, exc
                )
                time.sleep(delay * (2 ** i))

        detail = f"{type(last).__name__}: {last}"
        if _is_transient(last):
            breaker.on_failure()
            raise VectorStoreUnavailable(
                "向量检索服务暂时不可用，请稍后重试",
                detail=detail,
            ) from last
        # 请求本身有问题（4xx / 配置错），不计入熔断：重试和熔断都救不了它
        raise VectorStoreUnavailable(
            f"向量库请求被拒绝（{op}），通常是配置或请求本身有问题",
            detail=detail,
        ) from last

    # ------------------------------------------------------------------ #
    def _ensure_collection(self) -> None:
        from qdrant_client import models

        rk = (settings.qdrant_url, self._collection)
        with _LOCK:
            ready_dim = _READY.get(rk)
        if ready_dim == self._dim:
            return

        existing = {
            c.name for c in self._call("get_collections", self._client.get_collections).collections
        }

        if self._collection in existing:
            actual = self._collection_dim()
            if actual is not None and actual != self._dim:
                raise DimensionMismatch(
                    f"Qdrant 集合 '{self._collection}' 的向量维度是 {actual}，"
                    f"但当前 embedding 维度是 {self._dim}。\n"
                    f"换过向量模型就必须重建集合并重新入库（不同模型的向量不可混用）。修法二选一：\n"
                    f"  1) 换个集合名：QDRANT_COLLECTION=aics_chunks_{self._dim}\n"
                    f"  2) 删掉旧集合并让系统自动重建（会清空已入库向量，需重新上传文档）：\n"
                    f"     在 Qdrant 控制台删除集合 '{self._collection}'，或在系统内重新索引所有文档。"
                )
        else:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
            logger.info("已创建 Qdrant 集合 %s（dim=%s）", self._collection, self._dim)

        # 索引必须补建，且**集合已存在时也要补**：Qdrant 用一个没有索引的字段做过滤，
        # 会直接返回 400（"Index required but not found for ..."）而不是降级成慢查询。
        # 老版本代码只建了 tenant_id/kb_id，导致按 doc_id 删除文档在真机会 400。
        self._ensure_indexes()

        with _LOCK:
            _READY[rk] = self._dim

    def _ensure_indexes(self) -> None:
        """为参与过滤的字段建 keyword 索引（幂等）。索引缺失 = 过滤直接 400。"""
        from qdrant_client import models

        for field in ("tenant_id", "kb_id", "doc_id"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                # 已存在等非致命错误：过滤正确性不受影响，但可能退化为慢查询
                logger.warning("创建 payload 索引 %s 未成功（不影响隔离正确性）：%s", field, exc)

    def _collection_dim(self) -> int | None:
        try:
            info = self._call(
                "get_collection", self._client.get_collection, self._collection
            )
            vec = info.config.params.vectors
        except VectorStoreUnavailable:
            raise
        except Exception:  # pragma: no cover
            return None
        if hasattr(vec, "size"):
            return int(vec.size)
        if isinstance(vec, dict):  # 多命名向量：取默认向量
            default = vec.get("") or next(iter(vec.values()), None)
            if default is not None and hasattr(default, "size"):
                return int(default.size)
        return None

    # ------------------------------------------------------------------ #
    def upsert(self, items: list[VectorItem]) -> int:
        from qdrant_client import models

        if not items:
            return 0
        points = []
        for it in items:
            vec = l2_normalize(it.vector)
            if len(vec) != self._dim:
                raise DimensionMismatch(
                    f"待写入向量维度 {len(vec)} 与集合维度 {self._dim} 不一致，已中止写入"
                )
            points.append(
                models.PointStruct(
                    id=_point_id(it.id),
                    vector=vec,
                    payload={
                        "chunk_id": it.id,
                        "tenant_id": it.tenant_id,
                        "kb_id": it.kb_id,
                        "doc_id": it.doc_id,
                    },
                )
            )
        self._call(
            "upsert",
            self._client.upsert,
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        return len(points)

    @staticmethod
    def _make_filter(tenant_id: str, kb_ids: list[str] | None, doc_ids: list[str] | None = None):
        from qdrant_client import models

        must = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        ]
        if kb_ids:
            must.append(models.FieldCondition(key="kb_id", match=models.MatchAny(any=list(kb_ids))))
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
        return models.Filter(must=must)

    def search(
        self,
        tenant_id: str,
        vector: list[float],
        *,
        top_k: int,
        kb_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorHit]:
        if not tenant_id:
            raise ValueError("search 必须携带 tenant_id")
        qfilter = self._make_filter(tenant_id, kb_ids, doc_ids)
        query_vec = l2_normalize(vector)

        # 兼容 qdrant-client 新旧 API
        if hasattr(self._client, "query_points"):
            res = self._call(
                "query_points",
                self._client.query_points,
                collection_name=self._collection,
                query=query_vec,
                limit=max(1, top_k),
                query_filter=qfilter,
                score_threshold=min_score if min_score > 0 else None,
                with_payload=True,
            )
            points = getattr(res, "points", res)
        else:  # pragma: no cover
            points = self._call(
                "search",
                self._client.search,
                collection_name=self._collection,
                query_vector=query_vec,
                limit=max(1, top_k),
                query_filter=qfilter,
                score_threshold=min_score if min_score > 0 else None,
                with_payload=True,
            )

        hits: list[VectorHit] = []
        for p in points:
            payload = p.payload or {}
            # 二次校验：即便过滤失效也不允许跨租户返回
            if payload.get("tenant_id") != tenant_id:
                continue
            hits.append(VectorHit(chunk_id=str(payload.get("chunk_id") or p.id), score=float(p.score)))
        return hits

    # ------------------------------------------------------------------ #
    def _delete(self, tenant_id: str, extra: dict) -> int:
        from qdrant_client import models

        must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
        for key, value in extra.items():
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        self._call(
            "delete",
            self._client.delete,
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )
        return -1  # Qdrant 不返回删除条数

    def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        return self._delete(tenant_id, {"doc_id": doc_id})

    def delete_by_kb(self, tenant_id: str, kb_id: str) -> int:
        return self._delete(tenant_id, {"kb_id": kb_id})

    def delete_tenant(self, tenant_id: str) -> int:
        return self._delete(tenant_id, {})

    # ------------------------------------------------------------------ #
    def healthcheck(self) -> dict:
        """启动自检用：确认可达、集合就绪、维度一致。不抛异常，把结论交给调用方。"""
        try:
            n = self._call(
                "count", self._client.count, collection_name=self._collection, exact=True
            ).count
        except Exception as exc:
            detail = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
            # 连不上时也要把「系统期望什么」说清楚：集群不可达恰恰是运维最需要
            # 核对目标集合名与期望维度的时刻（对不上就是另一类故障，别混在一起看）。
            return {
                "ok": False,
                "backend": "qdrant",
                "collection": self._collection,
                "dim": self._dim,
                "error": detail,
            }
        return {
            "ok": True,
            "backend": "qdrant",
            "collection": self._collection,
            "dim": self._dim,
            "points": n,
        }
