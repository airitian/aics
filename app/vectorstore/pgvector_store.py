"""VECTOR_BACKEND=pgvector：与业务库同源，运维最简单。

需要：PostgreSQL 且已安装 vector 扩展（CREATE EXTENSION vector）。
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.vectorstore.base import VectorHit, VectorItem, VectorStore, l2_normalize


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, db: Session, dim: int):
        self.db = db
        self.table = settings.pgvector_table
        self.dim = dim
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self.db.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id  varchar(32) PRIMARY KEY,
                    tenant_id varchar(32) NOT NULL,
                    kb_id     varchar(32) NOT NULL,
                    doc_id    varchar(32) NOT NULL,
                    embedding vector({self.dim})
                )
                """
            )
        )
        self.db.execute(
            text(f"CREATE INDEX IF NOT EXISTS ix_{self.table}_tenant ON {self.table} (tenant_id, kb_id)")
        )
        self.db.commit()

    def upsert(self, items: list[VectorItem]) -> int:
        if not items:
            return 0
        sql = text(
            f"""
            INSERT INTO {self.table} (chunk_id, tenant_id, kb_id, doc_id, embedding)
            VALUES (:chunk_id, :tenant_id, :kb_id, :doc_id, CAST(:embedding AS vector))
            ON CONFLICT (chunk_id) DO UPDATE
               SET kb_id = EXCLUDED.kb_id,
                   doc_id = EXCLUDED.doc_id,
                   embedding = EXCLUDED.embedding
             WHERE {self.table}.tenant_id = EXCLUDED.tenant_id
            """
        )
        for it in items:
            self.db.execute(
                sql,
                {
                    "chunk_id": it.id,
                    "tenant_id": it.tenant_id,
                    "kb_id": it.kb_id,
                    "doc_id": it.doc_id,
                    "embedding": _vec_literal(l2_normalize(it.vector)),
                },
            )
        return len(items)

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
        literal = _vec_literal(l2_normalize(vector))
        params: dict = {"tid": tenant_id, "vec": literal, "limit": max(1, top_k)}
        kb_clause = ""
        if kb_ids:
            kb_clause = "AND kb_id = ANY(:kbs)"
            params["kbs"] = list(kb_ids)
        doc_clause = ""
        if doc_ids:
            doc_clause = "AND doc_id = ANY(:docs)"
            params["docs"] = list(doc_ids)
        sql = text(
            f"""
            SELECT chunk_id, 1 - (embedding <=> CAST(:vec AS vector)) AS score
              FROM {self.table}
             WHERE tenant_id = :tid {kb_clause} {doc_clause}
             ORDER BY embedding <=> CAST(:vec AS vector)
             LIMIT :limit
            """
        )
        rows = self.db.execute(sql, params).all()
        return [
            VectorHit(chunk_id=r[0], score=float(r[1]))
            for r in rows
            if float(r[1]) >= min_score
        ]

    def _delete(self, tenant_id: str, where: str, params: dict) -> int:
        res = self.db.execute(
            text(f"DELETE FROM {self.table} WHERE tenant_id = :tid {where}"),
            {"tid": tenant_id, **params},
        )
        return int(res.rowcount or 0)

    def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        return self._delete(tenant_id, "AND doc_id = :doc", {"doc": doc_id})

    def delete_by_kb(self, tenant_id: str, kb_id: str) -> int:
        return self._delete(tenant_id, "AND kb_id = :kb", {"kb": kb_id})

    def delete_tenant(self, tenant_id: str) -> int:
        return self._delete(tenant_id, "", {})
