"""向量库适配层接口。

关键设计：tenant_id 是 search / delete 的**必填位置参数**。
调用方不可能「忘记加租户过滤」—— 类型系统会拦住它。这是把租户隔离
从「约定」变成「编译期约束」的做法（PRD 2.3）。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VectorItem:
    id: str
    tenant_id: str
    kb_id: str
    doc_id: str
    text: str
    vector: list[float] = field(default_factory=list)


@dataclass
class VectorHit:
    chunk_id: str
    score: float


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return vector
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


class VectorStoreUnavailable(Exception):
    """向量库暂时不可用（网络抖动、集群重启、鉴权/配置错误）。

    单独定义这个类型，是为了让上层能把「向量库挂了」和「代码 bug」区分开：
    - 前者映射成 503（可恢复），并让对话走降级话术而不是编造答案；
    - 后者应该暴露成 500 让人去修。
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


class VectorStore(ABC):
    """所有实现都必须把 tenant_id 作为检索的第一道过滤条件。"""

    name = "base"

    @abstractmethod
    def upsert(self, items: list[VectorItem]) -> int:
        raise NotImplementedError

    @abstractmethod
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
        """返回本租户（且限定在 kb_ids / doc_ids 内）的命中，禁止跨租户召回。"""
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_by_kb(self, tenant_id: str, kb_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> int:
        raise NotImplementedError

    def healthcheck(self) -> dict:
        """启动自检：确认后端可达、可用。子类可覆写做更深的检查。"""
        return {"ok": True, "backend": self.name}
