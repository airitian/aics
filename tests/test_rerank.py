"""重排（Rerank）回归测试：二阶段检索的粗排放宽、精排生效、降级收紧。

离线测试，不联网。用假 store + 假 rerank 锁定路由与降级逻辑。
真机效果用 `python verify_retrieval.py` 测（Amazon 手册 75 题）：
启用后 R@1 从 69.3% 提到 82.7%。
"""
from __future__ import annotations

import asyncio

import pytest

from app import rag
from app.config import settings


@pytest.fixture(autouse=True)
def _pin_legacy_engine(monkeypatch):
    """本文件锁定的是 legacy 引擎的重排/降级行为；引擎开关切到 llama 时也不受影响。"""
    monkeypatch.setattr(settings, "rag_engine", "legacy")


class _Hit:
    def __init__(self, chunk_id: str, score: float, kb_id: str = "kbA"):
        self.chunk_id = chunk_id
        self.kb_id = kb_id
        self.score = score


class _FakeStore:
    def __init__(self, hits: list[_Hit]):
        self.hits = hits
        self.calls: list[int] = []

    def search(self, tenant_id, vector, *, top_k, kb_ids=None, min_score=0.0):
        self.calls.append(top_k)
        self.last_min_score = min_score
        return self.hits[:top_k]


class _Chunk:
    def __init__(self, id: str, text: str):
        self.id = id
        self.kb_id = "kbA"
        self.doc_id = "doc-" + id
        self.text = text


class _Row:
    def __init__(self, chunk: _Chunk, filename: str):
        self.chunk = chunk
        self.filename = filename

    def __getitem__(self, i):
        return (self.chunk, self.filename)[i]


def _install(monkeypatch, store: _FakeStore, chunks: list[_Chunk], *, rerank_on=True, scores=None):
    # 固定阈值：不显式钉住就会继承测试环境的 settings，断言变成碰运气
    monkeypatch.setattr(settings, "min_score", 0.45)
    monkeypatch.setattr(settings, "rerank_min_score", 0.05)
    monkeypatch.setattr(settings, "rerank_recall_min_score", 0.15)

    async def fake_embed_query(q):
        return [0.1, 0.2], 7

    monkeypatch.setattr(rag, "embed_query", fake_embed_query)
    monkeypatch.setattr(rag, "build_vector_store", lambda db, dim=None: store)
    monkeypatch.setattr(rag, "rerank_enabled", lambda: rerank_on)

    async def fake_rerank(query, docs):
        return None if scores is None else list(scores)

    monkeypatch.setattr(rag, "rerank_passages", fake_rerank)

    class _FakeDB:
        def __init__(self):
            self.added: list[object] = []

        def execute(self, stmt):
            rows = [_Row(c, f"f-{c.id}") for c in chunks]

            class _R:
                def all(self):
                    return rows

                def scalars(self):
                    raise AssertionError("retrieve 应取 (Chunk, filename) 元组")

            return _R()

        # 重排成功时会记一条 kind="rerank" 的用量
        def add(self, obj):
            self.added.append(obj)

        def flush(self):
            pass

        def commit(self):
            pass

    return _FakeDB()


def _chunks(n: int) -> list[_Chunk]:
    return [_Chunk(f"c{i}", f"内容{i}") for i in range(1, n + 1)]


def test_rerank_widens_recall(monkeypatch):
    """启用重排后，粗排必须放宽到 RERANK_CANDIDATES，否则答案常在第 6 名开外。"""
    store = _FakeStore([_Hit(f"c{i}", 0.9 - i * 0.01) for i in range(1, 21)])
    db = _install(monkeypatch, store, _chunks(20), scores=[0.9] * 20)

    hits, _ = asyncio.run(rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q"))
    assert store.calls == [settings.rerank_candidates * 3]  # 粗取放大 3 倍
    # 粗排门槛取「原阈值与 RERANK_RECALL_MIN_SCORE 的较小者」：
    # 粗排只负责别漏，噪声交给精排筛。
    assert store.last_min_score == min(0.45, settings.rerank_recall_min_score)
    assert store.last_min_score < 0.45
    # 精排后仍只返回 top_k 条给提示词
    assert len(hits) == settings.top_k


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    """精排分与向量分不一致时，以精排分为准（这是 rerank 的全部意义）。"""
    store = _FakeStore([
        _Hit("c1", 0.90),   # 向量分最高
        _Hit("c2", 0.50),   # 向量分垫底，但精排判它最相关
        _Hit("c3", 0.60),
    ])
    # c1 向量分最高但精排最低；三者都高于 RERANK_MIN_SCORE，只考排序不考过滤
    db = _install(monkeypatch, store, _chunks(3), scores=[0.20, 0.95, 0.30])

    hits, _ = asyncio.run(rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q"))
    assert [h.chunk_id for h in hits] == ["c2", "c3", "c1"]
    assert hits[0].score == 0.95, "score 应被覆盖为精排分，供下游排序与展示"


def test_rerank_drops_off_topic(monkeypatch):
    """低于 RERANK_MIN_SCORE 的候选不进提示词 —— 宁可少给，也不灌噪声。"""
    store = _FakeStore([_Hit("c1", 0.90), _Hit("c2", 0.80), _Hit("c3", 0.70)])
    db = _install(monkeypatch, store, _chunks(3), scores=[0.95, 0.10, 0.02])
    # 必须放在 _install 之后：_install 会先把阈值钉成默认值
    monkeypatch.setattr(settings, "rerank_min_score", 0.5)

    hits, _ = asyncio.run(rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q"))
    assert [h.chunk_id for h in hits] == ["c1"]


def test_rerank_unavailable_falls_back_and_tightens(monkeypatch):
    """重排挂了要降级，且**必须按原 MIN_SCORE 收紧**。

    粗排是放宽过的（门槛更低、条数更多），不收紧的话降级反而灌进更多噪声。
    """
    store = _FakeStore([_Hit("c1", 0.90), _Hit("c2", 0.60), _Hit("c3", 0.50)])
    # scores=None 模拟重排不可用
    db = _install(monkeypatch, store, _chunks(3), scores=None)
    # 放在 _install 之后：_install 会先把阈值钉成默认值
    monkeypatch.setattr(settings, "min_score", 0.75)

    hits, _ = asyncio.run(rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q"))
    assert [h.chunk_id for h in hits] == ["c1"], "降级后只剩过 MIN_SCORE 的候选"
    assert hits[0].score == 0.90, "降级时 score 保持向量分"


def test_rerank_off_uses_plain_vector_order(monkeypatch):
    """未启用重排时行为不变：候选原样返回，不触发任何重排调用。"""
    store = _FakeStore([_Hit("c1", 0.90), _Hit("c2", 0.80)])
    db = _install(monkeypatch, store, _chunks(2), rerank_on=False, scores=[0.99, 0.98])

    hits, _ = asyncio.run(rag.retrieve(db, tenant_id="t1", kb_ids=["kbA"], query="q"))
    assert store.calls == [settings.top_k * 3]
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
