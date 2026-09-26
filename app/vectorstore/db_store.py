"""VECTOR_BACKEND=db：复用关系库做向量检索（零额外依赖）。

向量以 JSON 存在 chunks.embedding，检索时先按 tenant_id + kb_id + enabled
过滤再算余弦 —— 这是最容易被写错的地方：**过滤必须发生在 SQL 层**，
不能先全量取回再在内存里筛，否则一旦有人改错就是跨租户召回。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.vectorstore.base import VectorHit, VectorItem, VectorStore, cosine, l2_normalize
from app.utils import j_dump, j_load


class DbVectorStore(VectorStore):
    name = "db"

    def __init__(self, db: Session):
        self.db = db

    def upsert(self, items: list[VectorItem]) -> int:
        if not items:
            return 0
        written = 0
        # 按租户分组：一次 upsert 只处理一个租户，避免用 items[0] 的租户去查别的租户的行
        by_tenant: dict[str, list[VectorItem]] = {}
        for it in items:
            by_tenant.setdefault(it.tenant_id, []).append(it)

        for tenant_id, group in by_tenant.items():
            ids = [it.id for it in group]
            existing = {
                row.id: row
                for row in self.db.execute(
                    select(Chunk).where(Chunk.id.in_(ids), Chunk.tenant_id == tenant_id)
                )
                .scalars()
                .all()
            }
            for it in group:
                row = existing.get(it.id)
                if row is not None and row.tenant_id != tenant_id:
                    # 主键撞上别的租户：绝不覆盖
                    continue
                vector = l2_normalize(it.vector)
                if row is None:
                    row = Chunk(
                        id=it.id,
                        tenant_id=it.tenant_id,
                        kb_id=it.kb_id,
                        doc_id=it.doc_id,
                        text=it.text,
                        embedding=j_dump(vector),
                    )
                    self.db.add(row)
                else:
                    row.kb_id = it.kb_id
                    row.doc_id = it.doc_id
                    row.text = it.text
                    row.embedding = j_dump(vector)
                written += 1
        return written

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
        query = l2_normalize(vector)
        stmt = select(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.enabled.is_(True),
            Chunk.embedding != "",
        )
        if kb_ids:
            stmt = stmt.where(Chunk.kb_id.in_(kb_ids))
        if doc_ids:
            stmt = stmt.where(Chunk.doc_id.in_(doc_ids))

        scored: list[VectorHit] = []
        for row in self.db.execute(stmt).scalars().all():
            vec = j_load(row.embedding, None)
            if not vec:
                continue
            score = cosine(query, vec)
            if score >= min_score:
                scored.append(VectorHit(chunk_id=row.id, score=round(score, 6)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(1, top_k)]

    def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        rows = self.db.execute(
            select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.doc_id == doc_id)
        ).scalars().all()
        for row in rows:
            self.db.delete(row)
        return len(rows)

    def delete_by_kb(self, tenant_id: str, kb_id: str) -> int:
        rows = self.db.execute(
            select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.kb_id == kb_id)
        ).scalars().all()
        for row in rows:
            self.db.delete(row)
        return len(rows)

    def delete_tenant(self, tenant_id: str) -> int:
        rows = self.db.execute(select(Chunk).where(Chunk.tenant_id == tenant_id)).scalars().all()
        for row in rows:
            self.db.delete(row)
        return len(rows)
