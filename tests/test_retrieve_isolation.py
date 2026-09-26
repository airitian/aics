"""检索隔离回归测试：多知识库按库独立检索，互不挤占。

背景：此前多库是合并进一次向量检索全局排序取 top_k，内容多的大库会把
小库的高分段整个挤出结果。现改为每库独立搜索、各取保底候选后合并。
本文件用假 store + 假 embedding 离线锁定路由逻辑，不联网。
"""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from app import rag
from app.config import settings


@pytest.fixture(autouse=True)
def _pin_legacy_engine(monkeypatch):
    """本文件锁定的是 legacy 引擎的路由逻辑；引擎开关切到 llama 时也不受影响。"""
    monkeypatch.setattr(settings, "rag_engine", "legacy")


class _Hit:
    def __init__(self, chunk_id: str, kb_id: str, score: float):
        self.chunk_id = chunk_id
        self.kb_id = kb_id
        self.score = score


class _FakeStore:
    """记录每次调用的 kb_ids / top_k，按预设返回命中。"""

    def __init__(self, results_by_kb: dict[str, list[_Hit]]):
        self.results_by_kb = results_by_kb
        self.calls: list[tuple[list[str], int]] = []

    def search(self, tenant_id, vector, *, top_k, kb_ids=None, min_score=0.0):
        assert tenant_id == "t1"
        assert len(kb_ids) == 1, "隔离检索必须一次只查一个库"
        kb = kb_ids[0]
        self.calls.append((kb, top_k))
        return self.results_by_kb.get(kb, [])[:top_k]


class _Chunk:
    def __init__(self, id: str, kb_id: str, text: str):
        self.id = id
        self.kb_id = kb_id
        self.doc_id = "doc-" + id
        self.text = text


class _Row:
    def __init__(self, chunk: _Chunk, filename: str):
        self.chunk = chunk
        self.filename = filename

    def __getitem__(self, i):
        return (self.chunk, self.filename)[i]


def _install(monkeypatch, store: _FakeStore, chunks: list[_Chunk]):
    async def fake_embed_query(q):
        return [0.1, 0.2], 7

    monkeypatch.setattr(rag, "embed_query", fake_embed_query)
    monkeypatch.setattr(rag, "build_vector_store", lambda db, dim=None: store)
    # 本文件锁定的是「多库隔离路由」，与重排无关。
    # 不关掉的话粗排条数会变成 RERANK_CANDIDATES，断言里写死的 top_k 就对不上了。
    monkeypatch.setattr(rag, "rerank_enabled", lambda: False)

    class _FakeDB:
        def execute(self, stmt):
            rows = [_Row(c, f"f-{c.id}") for c in chunks]
            class _R:
                def scalars(self):
                    raise AssertionError("retrieve 应取 (Chunk, filename) 元组")
                def all(self):
                    return rows
            return _R()

    return _FakeDB()


def test_single_kb_uses_full_top_k(monkeypatch):
    store = _FakeStore({"kbA": [_Hit("c1", "kbA", 0.9), _Hit("c2", "kbA", 0.8), _Hit("c3", "kbA", 0.7)]})
    chunks = [_Chunk("c1", "kbA", "a1"), _Chunk("c2", "kbA", "a2"), _Chunk("c3", "kbA", "a3")]
    db = _install(monkeypatch, store, chunks)

    hits, tokens = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q")
    )
    assert tokens == 7
    # 单库保持原行为：一次查询、取满 top_k
    assert store.calls == [("kbA", settings.top_k * 3)]  # 粗取放大 3 倍（抗标识符噪声块挤池）
    assert [h.chunk_id for h in hits] == ["c1", "c2", "c3"]


def test_multi_kb_searched_separately_and_merged(monkeypatch):
    # kbA 内容多、分数普遍高；kbB 只有一个 0.62 的段
    store = _FakeStore({
        "kbA": [_Hit("a1", "kbA", 0.91), _Hit("a2", "kbA", 0.88), _Hit("a3", "kbA", 0.61)],
        "kbB": [_Hit("b1", "kbB", 0.62)],
    })
    chunks = [_Chunk("a1", "kbA", "x1"), _Chunk("a2", "kbA", "x2"),
              _Chunk("a3", "kbA", "x3"), _Chunk("b1", "kbB", "y1")]
    db = _install(monkeypatch, store, chunks)

    hits, _ = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=["kbA", "kbB"], query="q")
    )
    # 每库各查一次，保底 = max(per_kb_top_k, ceil(top_k/2)) = 3
    assert sorted(store.calls) == [("kbA", 12), ("kbB", 12)]  # 保底4 × 3倍粗取
    # 合并后按分数排序，小库的 0.62 不会因为全局 top_k 被挤掉（0.61 的 a3 才垫底）
    assert [h.chunk_id for h in hits] == ["a1", "a2", "b1", "a3"]
    assert hits[2].kb_id == "kbB"


def test_multi_kb_per_kb_floor_is_two(monkeypatch):
    # 3 个库：保底 = max(2, ceil(5/3)) = 2
    store = _FakeStore({
        "kbA": [_Hit("a1", "kbA", 0.9), _Hit("a2", "kbA", 0.5)],
        "kbB": [_Hit("b1", "kbB", 0.8)],
        "kbC": [],
    })
    chunks = [_Chunk("a1", "kbA", "x"), _Chunk("a2", "kbA", "y"), _Chunk("b1", "kbB", "z")]
    db = _install(monkeypatch, store, chunks)

    hits, _ = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=["kbA", "kbB", "kbC"], query="q")
    )
    assert sorted(store.calls) == [("kbA", 9), ("kbB", 9), ("kbC", 9)]  # 保底3 × 3倍粗取
    assert [h.chunk_id for h in hits] == ["a1", "b1", "a2"]


def test_empty_kb_ids_returns_empty(monkeypatch):
    store = _FakeStore({})
    db = _install(monkeypatch, store, [])
    hits, tokens = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=[], query="q")
    )
    assert hits == [] and tokens == 0
    assert store.calls == []


# --------------------------------------------------------------------------- #
# 文件级绑定（doc_ids）
# --------------------------------------------------------------------------- #
class _DocStore:
    """记录 doc_ids 过滤参数的假 store。"""

    def __init__(self, hits: list[_Hit]):
        self.hits = hits
        self.calls: list[tuple[list[str] | None, list[str] | None, int]] = []

    def search(self, tenant_id, vector, *, top_k, kb_ids=None, doc_ids=None, min_score=0.0):
        assert tenant_id == "t1"
        self.calls.append((kb_ids, doc_ids, top_k))
        pool = self.hits
        if doc_ids:
            pool = [h for h in pool if getattr(h, "doc_id", None) in set(doc_ids)]
        if kb_ids:
            pool = [h for h in pool if h.kb_id in set(kb_ids)]
        return sorted(pool, key=lambda h: h.score, reverse=True)[:top_k]


def test_doc_ids_take_precedence_over_kb_ids(monkeypatch):
    # 同时给了 doc_ids 和 kb_ids：必须按 doc_ids 全局排序取 top_k，不走按库隔离
    store = _DocStore([
        _Hit("c1", "kbA", 0.95),
        _Hit("c2", "kbA", 0.85),
        _Hit("c3", "kbB", 0.75),
        _Hit("c4", "kbB", 0.65),
    ])
    for i, h in enumerate(store.hits, 1):
        h.doc_id = f"doc{i}"
    chunks = [_Chunk(f"c{i}", "kbX", f"t{i}") for i in range(1, 5)]
    db = _install(monkeypatch, store, chunks)

    hits, tokens = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q", doc_ids=["doc1", "doc3", "doc4"])
    )
    assert tokens == 7
    assert store.calls == [(None, ["doc1", "doc3", "doc4"], settings.top_k * 3)]
    # 全局排序：c1(0.95) → c3(0.75) → c4(0.65)，c2 因不在 doc 范围被过滤
    assert [h.chunk_id for h in hits] == ["c1", "c3", "c4"]


def test_doc_ids_empty_falls_back_to_kb(monkeypatch):
    store = _DocStore([_Hit("c1", "kbA", 0.9)])
    chunks = [_Chunk("c1", "kbA", "t")]
    db = _install(monkeypatch, store, chunks)

    hits, _ = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q", doc_ids=[])
    )
    assert store.calls == [(["kbA"], None, settings.top_k * 3)]
    assert [h.chunk_id for h in hits] == ["c1"]


def test_both_empty_returns_no_hits(monkeypatch):
    store = _DocStore([_Hit("c1", "kbA", 0.9)])
    db = _install(monkeypatch, store, [])
    hits, tokens = asyncio.run(
        rag.retrieve(db, tenant_id="t1", kb_ids=[], query="q", doc_ids=[])
    )
    assert hits == [] and tokens == 0
    assert store.calls == []


# --------------------------------------------------------------------------- #
# 主键精确命中（订单号 / 运单号）
#
# 真实踩坑：「订单 5127693301594110026 的运单号」——编号没有语义，向量粗排被一堆
# 同字段但**别的订单**的片段挤掉，正确那条没进 top5，模型于是答"没有记录"。
# 精确匹配是确定性的，命中就必须进上下文。
# --------------------------------------------------------------------------- #
def test_long_id_query_pins_exact_chunk(client, star):
    """提问含长编号时，必须锁定原文包含该编号的片段，且排在语义命中之前。"""
    import asyncio

    from app.database import SessionLocal
    from app.models import Chunk
    from app.rag import _pinned_key_chunks

    kb = client.post("/api/knowledge-bases", headers=star,
                     json={"name": "订单库"}).json()
    body = ("订单编号：5127693301594110026；实付款(元)：30.88；"
            "运单号：JT3174734445839；物流公司：极兔速递")
    res = client.post(f"/api/knowledge-bases/{kb['id']}/documents", headers=star,
                      files={"files": ("订单.txt", body.encode("utf-8"), "text/plain")})
    assert res.status_code == 200, res.text

    db = SessionLocal()
    try:
        row = db.execute(
            select(Chunk).where(Chunk.kb_id == kb["id"])
        ).scalars().first()
        assert row is not None, "文档应已入库"

        pinned = _pinned_key_chunks(
            db, tenant_id=row.tenant_id,
            query="帮我查一下订单 5127693301594110026 的运单号",
            kb_ids=[kb["id"]], doc_ids=[],
        )
        assert pinned, "长编号提问必须精确命中"
        assert "5127693301594110026" in pinned[0].text
        assert "JT3174734445839" in pinned[0].text
        assert pinned[0].score >= 1.0, "精确命中要压过语义分"
    finally:
        db.close()


def test_no_id_query_pins_nothing(client, star):
    """普通提问不含长编号时不触发精确匹配，避免把无关片段硬塞进上下文。"""
    from app.database import SessionLocal
    from app.rag import _pinned_key_chunks

    kb = client.post("/api/knowledge-bases", headers=star,
                     json={"name": "普通库"}).json()
    res = client.post(f"/api/knowledge-bases/{kb['id']}/documents", headers=star,
                      files={"files": ("资料.txt", ("无线网桥 WB730 参数与价格说明。" * 20).encode("utf-8"),
                                       "text/plain")})
    assert res.status_code == 200, res.text

    db = SessionLocal()
    try:
        pinned = _pinned_key_chunks(
            db, tenant_id="任意租户", query="无线网桥哪款最好用",
            kb_ids=[kb["id"]], doc_ids=[],
        )
        assert pinned == []
    finally:
        db.close()
