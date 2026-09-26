"""Qdrant 后端的离线回归测试（不联网）。

这些用例锁住两个只有在**真实 Qdrant 上**才会暴露的坑，本地 `db` 后端永远测不出来：

1. **过滤字段必须先建索引**：Qdrant 拿一个没索引的字段做过滤，不会降级成慢查询，
   而是直接回 400 `Index required but not found for "doc_id"`。
   原实现只给 tenant_id/kb_id 建了索引，导致「删除某篇文档」在生产直接 500。
2. **集合已存在时也要补建索引**：老集合是在上一版代码里建的，缺 doc_id 索引；
   如果 `_ensure_collection` 在"集合已存在"分支直接 return，老集合永远补不上索引。

3. **维度不匹配必须炸**：换 embedding 模型后若沿用旧集合，检索结果无意义。
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.vectorstore import qdrant_store as qs
from app.vectorstore.base import VectorItem, VectorStoreUnavailable
from app.vectorstore.qdrant_store import DimensionMismatch, QdrantVectorStore, _is_transient


class _Collections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n})() for n in names]


class _FakeClient:
    """记录所有调用，用于断言「建了哪些索引」「建集合时用了什么维度」。"""

    def __init__(self, existing: dict[str, int] | None = None):
        self.existing = dict(existing or {})  # name -> dim
        self.created_collections: list[tuple[str, int]] = []
        self.created_indexes: list[tuple[str, str]] = []
        self.deleted_filters: list[object] = []
        self.upserted: list[object] = []
        self.query_points_result: list = []
        self.fail_index_for: set[str] = set()
        self.count_value = 0

    # --- 元数据 ---
    def get_collections(self):
        return _Collections(list(self.existing))

    def get_collection(self, name):
        dim = self.existing[name]
        vectors = type("V", (), {"size": dim})()
        params = type("P", (), {"vectors": vectors})()
        config = type("Cfg", (), {"params": params})()
        return type("Info", (), {"config": config})()

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append((collection_name, vectors_config.size))
        self.existing[collection_name] = vectors_config.size

    def create_payload_index(self, collection_name, field_name, field_schema, wait=False):
        if field_name in self.fail_index_for:
            raise RuntimeError("boom")
        self.created_indexes.append((collection_name, field_name))

    # --- 数据面 ---
    def upsert(self, collection_name, points, wait=False):
        self.upserted.append(points)

    def query_points(self, **kwargs):
        return type("R", (), {"points": self.query_points_result})()

    def search(self, **kwargs):  # pragma: no cover - 仅老版本客户端分支
        return self.query_points_result

    def delete(self, collection_name, points_selector, wait=False):
        self.deleted_filters.append(points_selector)

    def count(self, collection_name, exact=False):
        return type("Cnt", (), {"count": self.count_value})()


@pytest.fixture(autouse=True)
def _clean_caches(monkeypatch):
    """模块级缓存会串测试，每个用例前清空。"""
    qs._CLIENTS.clear()
    qs._READY.clear()
    qs._BREAKERS.clear()
    monkeypatch.setattr(settings, "qdrant_url", "http://fake-qdrant:6333", raising=False)
    monkeypatch.setattr(settings, "qdrant_api_key", "fake-key", raising=False)
    monkeypatch.setattr(settings, "qdrant_collection", "unit_coll", raising=False)
    yield
    qs._CLIENTS.clear()
    qs._READY.clear()
    qs._BREAKERS.clear()


def _install(monkeypatch, client: _FakeClient) -> None:
    """把 QdrantVectorStore.__init__ 换成不联网的版本，其余逻辑（含 _ensure_collection）保持真实。"""
    monkeypatch.setattr(
        "app.vectorstore.qdrant_store.QdrantVectorStore.__init__",
        lambda self, dim, _c=client: _init_with(self, dim, _c),
        raising=True,
    )


def _init_with(self, dim: int, client: _FakeClient) -> None:
    """复刻真实 __init__，但用假客户端，避免联网。"""
    self._client = client
    self._collection = settings.qdrant_collection
    self._dim = dim
    self._ensure_collection()


def test_new_collection_indexes_tenant_kb_and_doc(monkeypatch):
    """新建集合时，参与过滤的三个字段都必须建索引（doc_id 就是漏掉过的那个）。"""
    client = _FakeClient()
    _install(monkeypatch, client)
    QdrantVectorStore(512)

    assert client.created_collections == [("unit_coll", 512)]
    indexed = {f for _, f in client.created_indexes}
    assert indexed == {"tenant_id", "kb_id", "doc_id"}, f"实际建了 {indexed}"


def test_existing_collection_still_gets_missing_index(monkeypatch):
    """集合已存在（老版本建的、缺 doc_id 索引）时，仍要补建索引。

    这是回归重点：若 _ensure_collection 在"已存在"分支直接 return，
    老集合永远补不上 doc_id 索引，按文档删除就会一直 400。
    """
    client = _FakeClient(existing={"unit_coll": 512})
    _install(monkeypatch, client)
    QdrantVectorStore(512)

    assert client.created_collections == [], "集合已存在就不该重复创建"
    indexed = {f for _, f in client.created_indexes}
    assert "doc_id" in indexed, "已存在的集合也必须补建 doc_id 索引"


def test_index_failure_is_non_fatal_but_logged(monkeypatch, caplog):
    """建索引失败不应让应用起不来，但必须留下告警。"""
    client = _FakeClient()
    client.fail_index_for = {"doc_id"}
    _install(monkeypatch, client)
    with caplog.at_level("WARNING"):
        QdrantVectorStore(512)
    assert "doc_id" in caplog.text


def test_dimension_mismatch_raises_with_actionable_message(monkeypatch):
    """换 embedding 模型后沿用旧集合，必须立刻报错并给出修法。"""
    client = _FakeClient(existing={"unit_coll": 1536})
    _install(monkeypatch, client)
    with pytest.raises(DimensionMismatch) as ei:
        QdrantVectorStore(512)
    msg = str(ei.value)
    assert "1536" in msg and "512" in msg
    assert "QDRANT_COLLECTION" in msg, "要给出可执行修法，而不是只报数字"


def test_upsert_rejects_wrong_dimension(monkeypatch):
    """待写入向量维度不对时中止写入，避免把垃圾塞进集合。"""
    client = _FakeClient()
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    with pytest.raises(DimensionMismatch):
        store.upsert(
            [VectorItem(id="a" * 32, tenant_id="t", kb_id="k", doc_id="d", text="x", vector=[0.1] * 8)]
        )


def test_search_requires_tenant_id(monkeypatch):
    """search 少了 tenant_id 必须直接报错，不允许退化成全库检索。"""
    client = _FakeClient()
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    with pytest.raises(ValueError):
        store.search("", [0.1] * 512, top_k=5)


def test_search_drops_cross_tenant_point_forced_in(monkeypatch):
    """即使向量库返回了别人的点（过滤被绕过/配错），二次校验也必须丢掉。"""
    client = _FakeClient()
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    client.query_points_result = [
        type("P", (), {"id": "1", "score": 0.99, "payload": {"tenant_id": "OTHER", "chunk_id": "x"}})(),
        type("P", (), {"id": "2", "score": 0.90, "payload": {"tenant_id": "mine", "chunk_id": "y"}})(),
    ]
    hits = store.search("mine", [0.1] * 512, top_k=5)
    assert [h.chunk_id for h in hits] == ["y"], "别人的点在二次校验里必须被丢掉"


def test_delete_by_doc_filters_on_doc_id(monkeypatch):
    """按文档删除必须真的带上 doc_id 条件（配合索引才有意义）。"""
    client = _FakeClient()
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    store.delete_by_doc("t1", "doc-1")
    assert len(client.deleted_filters) == 1
    flt = client.deleted_filters[0].filter
    keys = {c.key: c.match.value for c in flt.must}
    assert keys == {"tenant_id": "t1", "doc_id": "doc-1"}


def test_healthcheck_reports_dim_and_points(monkeypatch):
    client = _FakeClient(existing={"unit_coll": 512})
    client.count_value = 7
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    hc = store.healthcheck()
    assert hc == {"ok": True, "backend": "qdrant", "collection": "unit_coll", "dim": 512, "points": 7}


# --------------------------------------------------------------------------- #
# 网络抖动：必须重试，且必须区分「可重试」与「请求写错了」
# --------------------------------------------------------------------------- #
class _FlakyClient(_FakeClient):
    """前 N 次数据面调用抛网络错误，之后成功。用于验证退避重试。

    注意只让**数据面**（count/query_points）抖动，元数据面（get_collections）保持正常，
    否则构造期就挂了，测不到「重试吃掉抖动」这件事。
    """

    def __init__(self, fail_times: int, exc_factory):
        super().__init__(existing={"unit_coll": 512})
        self.fail_left = fail_times
        self.exc_factory = exc_factory
        self.calls = 0
        self.count_value = 3

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_left > 0:
            self.fail_left -= 1
            raise self.exc_factory()

    def count(self, collection_name, exact=False):
        self._maybe_fail()
        return type("Cnt", (), {"count": self.count_value})()

    def query_points(self, **kwargs):
        self._maybe_fail()
        return type("R", (), {"points": []})()


def _net_error():
    from qdrant_client.http.exceptions import ResponseHandlingException

    return ResponseHandlingException(
        OSError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """重试退避不真的睡，否则测试慢。"""
    monkeypatch.setattr("app.vectorstore.qdrant_store.time.sleep", lambda *_: None)


def test_transient_error_is_retried_then_succeeds(monkeypatch):
    """TLS 抖一下不应该让客户会话降级 —— 重试应当把它吃掉。"""
    client = _FlakyClient(fail_times=2, exc_factory=_net_error)
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    hc = store.healthcheck()
    assert hc["ok"] is True, hc
    assert client.calls == 3, f"应重试到成功，实际调用 {client.calls} 次"


def test_transient_error_exhausts_retries_and_raises_unavailable(monkeypatch):
    """一直抖就明确报「向量库不可用」，而不是抛出晦涩的底层异常。"""
    client = _FlakyClient(fail_times=99, exc_factory=_net_error)
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)  # 构造期也重试，这里集合已存在所以不会失败
    with pytest.raises(VectorStoreUnavailable):
        store.search("t", [0.1] * 512, top_k=3)
    assert client.calls == settings.qdrant_retries, f"应为 {settings.qdrant_retries} 次"


def test_client_error_is_not_retried(monkeypatch):
    """4xx 是请求本身写错了，重试只会把 bug 伪装成「不稳定」，必须立刻抛。"""
    from qdrant_client.http.exceptions import UnexpectedResponse

    class _BadRequestClient(_FakeClient):
        def __init__(self):
            super().__init__(existing={"unit_coll": 512})
            self.calls = 0

        def count(self, collection_name, exact=False):
            self.calls += 1
            raise UnexpectedResponse(
                status_code=400,
                reason_phrase="Bad Request",
                content=b'{"status":{"error":"Bad request"}}',
                headers={},
            )

    client = _BadRequestClient()
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    hc = store.healthcheck()
    assert hc["ok"] is False
    assert client.calls == 1, f"400 不应重试，实际调用 {client.calls} 次"


def test_is_transient_classification():
    """把「可重试」的判定单独锁住，避免以后放宽成「全都重试」。"""
    from qdrant_client.http.exceptions import UnexpectedResponse

    assert _is_transient(_net_error()) is True
    assert _is_transient(TimeoutError()) is True
    assert _is_transient(ConnectionError()) is True
    for code, expected in ((400, False), (401, False), (403, False), (404, False),
                           (429, True), (500, True), (503, True)):
        exc = UnexpectedResponse(
            status_code=code, reason_phrase="x", content=b"{}", headers={}
        )
        assert _is_transient(exc) is expected, f"status={code} 判定错误"


# --------------------------------------------------------------------------- #
# 熔断：向量库「真挂了」时不能每条消息都白等重试
# --------------------------------------------------------------------------- #
def test_breaker_opens_after_threshold_and_stops_hitting_network(monkeypatch):
    """连续失败到阈值后，后续请求必须直接失败，不再打网络、不再退避等待。

    这是客户体验问题：Qdrant 真挂了的时候，逐请求重试 = 每条消息白等 1.5 秒
    还照样答不上来，等于「又慢又没用」。
    """
    client = _FlakyClient(fail_times=10_000, exc_factory=_net_error)
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)

    # 构造期已用了 1 次 get_collections（成功），下面开始统计数据面调用
    client.calls = 0
    threshold = settings.qdrant_breaker_threshold

    for _ in range(threshold):
        with pytest.raises(VectorStoreUnavailable):
            store.search("t", [0.1] * 512, top_k=3)

    calls_after_trip = client.calls
    assert calls_after_trip == threshold * settings.qdrant_retries, (
        f"熔断前应各重试 {settings.qdrant_retries} 次，实际 {calls_after_trip}"
    )

    # 熔断已开：再问 5 次，网络调用次数不应增加
    for _ in range(5):
        with pytest.raises(VectorStoreUnavailable):
            store.search("t", [0.1] * 512, top_k=3)
    assert client.calls == calls_after_trip, (
        f"熔断后不应再打网络，实际从 {calls_after_trip} 涨到 {client.calls}"
    )


def test_breaker_recovers_after_cooldown(monkeypatch):
    """冷却时间过去后要放探针过去，恢复后正常返回结果。"""
    client = _FlakyClient(fail_times=10_000, exc_factory=_net_error)
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)

    for _ in range(settings.qdrant_breaker_threshold):
        with pytest.raises(VectorStoreUnavailable):
            store.search("t", [0.1] * 512, top_k=3)
    with pytest.raises(VectorStoreUnavailable):
        store.search("t", [0.1] * 512, top_k=3)

    # 让冷却时间过去，并让上游恢复
    client.fail_left = 0
    import time as _time

    real_monotonic = _time.monotonic  # 必须先取原函数，否则 lambda 会递归调用自己
    monkeypatch.setattr(
        "app.vectorstore.qdrant_store.time.monotonic",
        lambda: real_monotonic() + settings.qdrant_breaker_cooldown + 1,
    )
    hits = store.search("t", [0.1] * 512, top_k=3)
    assert hits == [], f"恢复后应正常返回（空结果），实际 {hits}"


def test_client_error_does_not_trip_breaker(monkeypatch):
    """4xx 是配置/请求错误，不该把熔断打开 —— 否则会把一个常量 bug 说成「服务不稳定」。"""
    from qdrant_client.http.exceptions import UnexpectedResponse

    class _BadClient(_FakeClient):
        def __init__(self):
            super().__init__(existing={"unit_coll": 512})

        def query_points(self, **kwargs):
            raise UnexpectedResponse(
                status_code=400, reason_phrase="Bad Request", content=b"{}", headers={}
            )

    _install(monkeypatch, _BadClient())
    store = QdrantVectorStore(512)
    for _ in range(settings.qdrant_breaker_threshold + 2):
        with pytest.raises(VectorStoreUnavailable):
            store.search("t", [0.1] * 512, top_k=3)
    assert qs._BREAKERS[("http://fake-qdrant:6333", "unit_coll")].is_open() is False


def test_healthcheck_failure_still_reports_expected_target(monkeypatch):
    """自检失败时也要给出「系统期望的目标」——集群连不上恰恰是运维最需要
    核对集合名与期望维度的时刻，否则只会看到一句「不可用」而不知该对什么。"""
    client = _FlakyClient(fail_times=10_000, exc_factory=_net_error)
    _install(monkeypatch, client)
    store = QdrantVectorStore(512)
    hc = store.healthcheck()
    assert hc["ok"] is False
    assert hc["collection"] == "unit_coll"
    assert hc["dim"] == 512
    assert hc["error"]


def test_startup_selfcheck_failure_keeps_expected_target(monkeypatch):
    """启动自检里「构造 store 就失败」的分支同样要带期望目标。

    集群不可达时走的就是这条分支（healthcheck 根本没机会执行），
    而运维恰恰要靠这里的集合名与期望维度判断该对什么。
    """
    from app import main as main_mod

    # 测试环境刻意把 VECTOR_BACKEND 锁成 db（防止误连真实云端），
    # 这里要验证的是 qdrant 分支的报错内容，所以显式切回来。
    monkeypatch.setattr(settings, "vector_backend", "qdrant", raising=False)

    def _boom(db):
        raise RuntimeError("connect failed")

    monkeypatch.setattr(main_mod, "build_vector_store", _boom)
    h = main_mod._check_vector_backend()
    assert h["ok"] is False
    assert h["collection"] == settings.qdrant_collection
    assert h["dim"] == settings.embed_dim
    assert "connect failed" in h["error"]
