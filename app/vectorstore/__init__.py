"""向量库工厂：按 VECTOR_BACKEND 选择实现。"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.vectorstore.base import (
    VectorHit,
    VectorItem,
    VectorStore,
    VectorStoreUnavailable,
    cosine,
    l2_normalize,
)
from app.vectorstore.db_store import DbVectorStore

__all__ = [
    "VectorStore",
    "VectorStoreUnavailable",
    "VectorItem",
    "VectorHit",
    "cosine",
    "l2_normalize",
    "build_vector_store",
]


def build_vector_store(db: Session, dim: int | None = None) -> VectorStore:
    backend = settings.vector_backend
    dim = dim or settings.embed_dim

    if backend == "db":
        return DbVectorStore(db)
    if backend == "qdrant":
        from app.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(dim)
    if backend == "pgvector":
        from app.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(db, dim)
    raise RuntimeError(f"未知的 VECTOR_BACKEND：{backend}")
