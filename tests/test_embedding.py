"""向量模型适配层的离线回归测试（不联网，用 MockTransport 打桩）。

锁住的是「配错/抖动」这几类只有接了真实远程模型才会暴露的问题：

1. **维度守卫**：bge-m3 是 1024 维。若 EMBED_DIM 配错（历史上默认是 512），
   向量会以错误维度写进 Qdrant，或者报一个隔了一层的 dimension mismatch。
   必须在模型这一层就当场炸，且 message 要给出修法。
2. **4xx 不重试也不熔断**：模型名写错 / Key 失效会返回 400/401。把这种
   「常量配置 bug」重试 3 次、再熔断 15 秒，等于把排查方向带偏成「服务不稳定」。
3. **5xx / 网络错要重试**：公网抖动不该让一次入库或一条客户消息直接失败。
4. **真故障要熔断**：连续失败到阈值后必须停止打网络，否则每条消息都白等退避。
5. **按 index 归位**：上游不保证顺序，错位会让向量和文本错配（且完全静默）。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app import embedding as emb
from app.config import settings
from app.embedding import (
    EmbeddingDimensionMismatch,
    EmbeddingUnavailable,
    embed_query,
    embed_texts,
)

DIM = 1024

# 必须在任何 monkeypatch 之前捕获：install 里若当场读 `httpx.AsyncClient`，
# 第二次 install 读到的会是上一次打桩后的工厂，transport 会被旧工厂覆盖回去。
_REAL_CLIENT = httpx.AsyncClient


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _api_provider(monkeypatch):
    """把适配层切到「远程 api」并清空模块级缓存，避免用例之间串状态。"""
    monkeypatch.setattr(settings, "embed_provider", "api", raising=False)
    monkeypatch.setattr(settings, "embed_base_url", "https://fake-embed/api/v1", raising=False)
    monkeypatch.setattr(settings, "embed_api_key", "fake-key", raising=False)
    monkeypatch.setattr(settings, "embed_model", "bge-m3", raising=False)
    monkeypatch.setattr(settings, "embed_dim", DIM, raising=False)
    monkeypatch.setattr(settings, "embed_batch", 32, raising=False)
    monkeypatch.setattr(settings, "embed_retries", 3, raising=False)
    # 退避不真的睡（delay=0 时 asyncio.sleep(0) 只是让出一次执行权），用例才不会慢
    monkeypatch.setattr(settings, "embed_retry_delay", 0.0, raising=False)
    monkeypatch.setattr(settings, "embed_breaker_threshold", 3, raising=False)
    monkeypatch.setattr(settings, "embed_breaker_cooldown", 15.0, raising=False)
    emb._CLIENTS.clear()
    emb._BREAKERS.clear()
    yield
    emb._CLIENTS.clear()
    emb._BREAKERS.clear()


def _ok_body(request: httpx.Request, dim: int = DIM, reverse: bool = False) -> dict:
    body = json.loads(request.content)
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i + 1)] * dim}
        for i in range(len(inputs))
    ]
    if reverse:
        data = list(reversed(data))
    return {
        "object": "list",
        "model": body.get("model"),
        "data": data,
        "usage": {"prompt_tokens": 7 * len(inputs), "total_tokens": 7 * len(inputs)},
    }


def install(monkeypatch, handler):
    """只替换底层 transport，**保留真实的 _client() 逻辑**。

    这样连接池复用、Bearer 头、端点拼接这些都在被测范围内；
    如果只把 `_client` 整个换掉，上面这些就都测不到了（测的成了打桩本身）。
    """
    calls: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    real_client = _REAL_CLIENT

    def _factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(_wrapped)
        return real_client(**kwargs)

    monkeypatch.setattr(emb.httpx, "AsyncClient", _factory)
    return calls


def http_error(request: httpx.Request, status: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, request=request)


# --------------------------------------------------------------------------- #
# 正常路径
# --------------------------------------------------------------------------- #
def test_returns_vectors_dim_and_tokens(monkeypatch):
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    res = run(embed_texts(["退货政策", "发货时效"]))
    assert len(res.vectors) == 2
    assert res.dim == DIM
    assert all(len(v) == DIM for v in res.vectors)
    assert res.tokens == 14
    assert res.provider == "api"


def test_vectors_are_realigned_by_index(monkeypatch):
    """上游把 data 倒着返回时，向量必须按 index 归位，不能按数组顺序直接用。"""
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req, reverse=True)))
    res = run(embed_texts(["甲", "乙", "丙"]))
    # 打桩里第 i 条的向量首元素是 i+1；若未按 index 归位，这里会是 3,2,1
    assert [v[0] for v in res.vectors] == [1.0, 2.0, 3.0]


def test_batch_splitting_preserves_order(monkeypatch):
    monkeypatch.setattr(settings, "embed_batch", 2, raising=False)
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    res = run(embed_texts(["a", "b", "c", "d", "e"]))
    sizes = [len(json.loads(c.content)["input"]) for c in calls]
    assert sizes == [2, 2, 1], f"应按 EMBED_BATCH 分批，实际 {sizes}"
    assert len(res.vectors) == 5


def test_blank_text_is_replaced_by_space(monkeypatch):
    """空串会让部分上游报错；应兜底成空格而不是把请求打回去。"""
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    run(embed_texts(["", "   "]))
    sent = json.loads(calls[0].content)["input"]
    assert sent == [" ", " "]


def test_empty_input_short_circuits(monkeypatch):
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    res = run(embed_texts([]))
    assert res.vectors == [] and calls == []


def test_embed_query_returns_single_vector(monkeypatch):
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    vec, tokens = run(embed_query("怎么退款"))
    assert len(vec) == DIM and tokens == 7


# --------------------------------------------------------------------------- #
# 维度守卫
# --------------------------------------------------------------------------- #
def test_dimension_mismatch_raises_actionable_error(monkeypatch):
    """模型实际输出维度与 EMBED_DIM 不符时，必须当场报错并给出修法。"""
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req, dim=512)))
    with pytest.raises(EmbeddingDimensionMismatch) as ei:
        run(embed_texts(["测试"]))
    msg = str(ei.value)
    assert "512" in msg and "1024" in msg
    assert "EMBED_DIM" in msg, "要给出可执行修法，而不是只报数字"
    assert len(calls) == 1, "配置错误不该重试"


def test_dimension_mismatch_not_tripping_breaker(monkeypatch):
    """维度配错是配置问题，不该把熔断打开（否则网络恢复后还要白等冷却）。"""
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req, dim=512)))
    for _ in range(settings.embed_breaker_threshold + 2):
        with pytest.raises(EmbeddingDimensionMismatch):
            run(embed_texts(["测试"]))
    assert emb._BREAKERS[("https://fake-embed/api/v1", "bge-m3")].is_open() is False


def test_count_mismatch_raises(monkeypatch):
    """返回条数与请求不一致时中止，避免向量与文本错配入库。"""

    def handler(req, n):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1] * DIM}], "usage": {}},
        )

    install(monkeypatch, handler)
    with pytest.raises(EmbeddingUnavailable) as ei:
        run(embed_texts(["a", "b"]))
    assert "条数" in str(ei.value)


# --------------------------------------------------------------------------- #
# 错误分流：4xx 立即抛 / 5xx 与网络错重试 / 真故障熔断
# --------------------------------------------------------------------------- #
def test_client_error_is_not_retried_and_surfaces_upstream_message(monkeypatch):
    """模型名写错返回 400 时必须立刻抛，且把上游原文带出来（否则只会看到一句「不可用」）。"""
    calls = install(
        monkeypatch,
        lambda req, n: http_error(req, 400, {"error": {"code": "400", "message": "模型名称“x”有误，请检查拼写"}}),
    )
    with pytest.raises(EmbeddingUnavailable) as ei:
        run(embed_texts(["测试"]))
    assert len(calls) == 1, f"400 不应重试，实际 {len(calls)} 次"
    assert "有误" in ei.value.detail, "上游错误原文要透出来，便于定位是模型名还是 Key"
    assert "HTTP 400" in str(ei.value)


def test_unauthorized_is_not_retried(monkeypatch):
    calls = install(monkeypatch, lambda req, n: http_error(req, 401, {"error": "invalid api key"}))
    with pytest.raises(EmbeddingUnavailable):
        run(embed_texts(["测试"]))
    assert len(calls) == 1


def test_client_error_does_not_trip_breaker(monkeypatch):
    install(monkeypatch, lambda req, n: http_error(req, 400, {"error": "bad model"}))
    for _ in range(settings.embed_breaker_threshold + 2):
        with pytest.raises(EmbeddingUnavailable):
            run(embed_texts(["测试"]))
    assert emb._BREAKERS[("https://fake-embed/api/v1", "bge-m3")].is_open() is False


def test_server_error_is_retried_then_succeeds(monkeypatch):
    """公网抖一下（503）应当被退避重试吃掉，客户无感。"""

    def handler(req, n):
        if n <= 2:
            return http_error(req, 503)
        return httpx.Response(200, json=_ok_body(req))

    calls = install(monkeypatch, handler)
    res = run(embed_texts(["退货"]))
    assert res.dim == DIM
    assert len(calls) == 3, f"应重试到成功，实际 {len(calls)} 次"


def test_network_error_is_retried(monkeypatch):
    def handler(req, n):
        if n <= 1:
            raise httpx.ConnectError("connection reset", request=req)
        return httpx.Response(200, json=_ok_body(req))

    calls = install(monkeypatch, handler)
    res = run(embed_texts(["退货"]))
    assert len(res.vectors) == 1
    assert len(calls) == 2


def test_transient_error_exhausts_retries(monkeypatch):
    calls = install(monkeypatch, lambda req, n: http_error(req, 503))
    with pytest.raises(EmbeddingUnavailable) as ei:
        run(embed_texts(["退货"]))
    assert len(calls) == settings.embed_retries, f"应为 {settings.embed_retries} 次"
    assert "暂时不可用" in str(ei.value)


def test_breaker_opens_and_stops_hitting_network(monkeypatch):
    """真故障（不是抖一下）时必须熔断：否则每条消息都要白等一轮退避。

    这是客户体验问题：模型挂了的时候，逐请求重试 = 又慢又照样答不上来。
    """
    calls = install(monkeypatch, lambda req, n: http_error(req, 503))
    threshold = settings.embed_breaker_threshold

    for _ in range(threshold):
        with pytest.raises(EmbeddingUnavailable):
            run(embed_texts(["退货"]))

    after_trip = len(calls)
    assert after_trip == threshold * settings.embed_retries

    for _ in range(4):
        with pytest.raises(EmbeddingUnavailable) as ei:
            run(embed_texts(["退货"]))
        assert "circuit breaker" in (ei.value.detail or "") or "不可用" in str(ei.value)
    assert len(calls) == after_trip, f"熔断后不应再打网络，实际从 {after_trip} 涨到 {len(calls)}"


def test_breaker_recovers_after_success(monkeypatch):
    """恢复一次成功后计数要清零，不能残留失败次数导致下次一抖就熔断。"""
    state = {"fail": True}

    def handler(req, n):
        if state["fail"]:
            return http_error(req, 503)
        return httpx.Response(200, json=_ok_body(req))

    install(monkeypatch, handler)
    for _ in range(settings.embed_breaker_threshold - 1):
        with pytest.raises(EmbeddingUnavailable):
            run(embed_texts(["退货"]))

    state["fail"] = False
    assert run(embed_texts(["退货"])).dim == DIM

    state["fail"] = True
    for _ in range(settings.embed_breaker_threshold - 1):
        with pytest.raises(EmbeddingUnavailable):
            run(embed_texts(["退货"]))
    assert emb._BREAKERS[("https://fake-embed/api/v1", "bge-m3")].is_open() is False


# --------------------------------------------------------------------------- #
# 配置缺失 / 异常响应
# --------------------------------------------------------------------------- #
def test_missing_config_raises_without_network(monkeypatch):
    monkeypatch.setattr(settings, "embed_base_url", "", raising=False)
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    with pytest.raises(EmbeddingUnavailable) as ei:
        run(embed_texts(["测试"]))
    assert "EMBED_BASE_URL" in str(ei.value)
    assert calls == [], "配置缺失时不该发请求"


def test_non_json_response_raises_clear_error(monkeypatch):
    install(
        monkeypatch,
        lambda req, n: httpx.Response(200, text="<html>nginx</html>", headers={"content-type": "text/html"}),
    )
    with pytest.raises(EmbeddingUnavailable) as ei:
        run(embed_texts(["测试"]))
    assert "EMBED_BASE_URL" in str(ei.value)


def test_auth_header_and_endpoint_are_correct(monkeypatch):
    """端点拼接必须是 <base>/embeddings，并带 Bearer 头（防回归）。"""
    calls = install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    run(embed_texts(["测试"]))
    assert str(calls[0].url) == "https://fake-embed/api/v1/embeddings"
    assert calls[0].headers["authorization"] == "Bearer fake-key"
    assert json.loads(calls[0].content)["model"] == "bge-m3"


# --------------------------------------------------------------------------- #
# local 兜底（离线开发/单测路径不能被改坏）
# --------------------------------------------------------------------------- #
def test_local_provider_is_deterministic_and_sized(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "local", raising=False)
    a = run(embed_texts(["退货政策"]))
    b = run(embed_texts(["退货政策"]))
    c = run(embed_texts(["发货时效"]))
    assert a.dim == DIM and len(a.vectors[0]) == DIM
    assert a.vectors[0] == b.vectors[0], "同一文本必须得到同一向量（否则检索不可复现）"
    assert a.vectors[0] != c.vectors[0]


# --------------------------------------------------------------------------- #
# 启动自检
# --------------------------------------------------------------------------- #
def test_healthcheck_reports_model_and_dim(monkeypatch):
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    hc = run(emb.healthcheck())
    assert hc["ok"] is True
    assert hc["dim"] == DIM
    assert hc["model"] == "bge-m3"
    assert hc["expected_dim"] == DIM


def test_healthcheck_failure_keeps_expected_target(monkeypatch):
    """自检失败时也要说清「系统期望什么」——维度配错正是最需要这句的时刻。"""
    install(monkeypatch, lambda req, n: http_error(req, 401, {"error": "invalid api key"}))
    hc = run(emb.healthcheck())
    assert hc["ok"] is False
    assert hc["expected_dim"] == DIM
    assert hc["model"] == "bge-m3"
    assert hc["error"]


def test_healthcheck_local_provider_is_ok_with_note(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "local", raising=False)
    hc = run(emb.healthcheck())
    assert hc["ok"] is True and "占位" in hc["note"]


def test_aclose_releases_clients(monkeypatch):
    """连接池必须能被释放，否则 uvicorn 退出时会报未关闭连接。"""
    install(monkeypatch, lambda req, n: httpx.Response(200, json=_ok_body(req)))
    run(embed_texts(["测试"]))
    assert emb._CLIENTS, "客户端应被缓存复用（而不是每批新建）"
    run(emb.aclose())
    assert emb._CLIENTS == {}
