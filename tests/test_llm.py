"""对话模型适配层的离线回归测试（不联网，用 MockTransport 打桩）。

锁住的是「接了真实推理模型之后才会暴露」的几类问题：

1. **配置错误不能伪装成服务抖动**（最重要）。
   实测 PackyAPI：模型名写错 → 返回 **503**（不是 404/400），真正的 `model_not_found`
   藏在 body 里。若只看状态码，它会落进「可重试」集合 —— 白等两轮退避、照样失败，
   最后报「模型暂时不可用」，把**配置写错了**说成**服务在抖动**，排查方向整个跑偏。
2. **推理模型的思维链绝不能发给客户**：`reasoning_content` 是模型的自言自语，
   里面有大量自我质疑和试错。一旦拼进回复发出去，等于当着客户的面自曝。
3. **截断不能静默**：思考 token 和正式回答共用 max_tokens 预算，撞上限时
   `finish_reason="length"`，客户收到半句话 —— 必须被标记出来。
4. **真故障要熔断**：模型侧持续报错时，不能每条客户消息都完整走完一轮退避。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app import llm
from app.config import settings
from app.llm import LLMConfigError, LLMUnavailable, chat, healthcheck

MODEL = "deepseek-v4-flash"

# 必须在任何 monkeypatch 之前捕获 —— install 里若当场读 `httpx.AsyncClient`，
# 第二次 install 读到的会是上一次打桩后的工厂，形成套娃：新 transport 被旧工厂覆盖回去，
# 表现为「换了 handler 但请求还是打到旧的」，极难看出是测试写错了。
_REAL_CLIENT = httpx.AsyncClient


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _api_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "api", raising=False)
    monkeypatch.setattr(settings, "llm_base_url", "https://fake-llm/v1", raising=False)
    monkeypatch.setattr(settings, "llm_api_key", "fake-key", raising=False)
    monkeypatch.setattr(settings, "llm_model", MODEL, raising=False)
    monkeypatch.setattr(settings, "llm_timeout", 5.0, raising=False)
    monkeypatch.setattr(settings, "llm_max_retries", 3, raising=False)
    monkeypatch.setattr(settings, "llm_temperature", 0.3, raising=False)
    monkeypatch.setattr(settings, "llm_max_tokens", 800, raising=False)
    monkeypatch.setattr(settings, "llm_breaker_threshold", 3, raising=False)
    monkeypatch.setattr(settings, "llm_breaker_cooldown", 15.0, raising=False)
    monkeypatch.setattr(settings, "llm_fallback_base_url", "", raising=False)
    monkeypatch.setattr(settings, "llm_fallback_api_key", "", raising=False)
    monkeypatch.setattr(settings, "llm_fallback_model", "", raising=False)
    # 退避不真的睡，否则用例会慢
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", _no_sleep)
    llm._CLIENTS.clear()
    llm._BREAKERS.clear()
    yield
    llm._CLIENTS.clear()
    llm._BREAKERS.clear()


def install(monkeypatch, handler):
    """只替换底层 transport，保留真实的 _client() 逻辑（连接池、Bearer 头都要在被测范围内）。"""
    calls: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    real_client = _REAL_CLIENT

    def _factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(_wrapped)
        return real_client(**kwargs)

    # 连接池是复用的：不清缓存的话，上一个用例的 client（带着旧 transport）会被一直用下去
    llm._CLIENTS.clear()
    monkeypatch.setattr(llm.httpx, "AsyncClient", _factory)
    return calls


def ok(content: str = "收到", include_reasoning: bool = False, finish="stop", usage=None):
    msg = {"role": "assistant", "content": content}
    if include_reasoning:
        msg["reasoning_content"] = "用户只说了你好。我不确定要不要直接回收到，先试试。"

    def handler(request: httpx.Request, _n: int) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }, request=request)

    return handler


def err(status: int, body):
    def handler(request: httpx.Request, _n: int) -> httpx.Response:
        return httpx.Response(status, json=body, request=request) if isinstance(body, dict) \
            else httpx.Response(status, text=body, request=request)

    return handler


MSGS = [{"role": "user", "content": "你好"}]


# --------------------------------------------------------------------------- #
# 正常路径
# --------------------------------------------------------------------------- #
def test_chat_returns_content_and_usage(monkeypatch):
    install(monkeypatch, ok("户外电源保修 24 个月"))
    r = run(chat(MSGS))
    assert r.text == "户外电源保修 24 个月"
    assert r.model == MODEL
    assert r.endpoint == "primary"
    assert r.prompt_tokens == 10 and r.completion_tokens == 5
    assert r.total_tokens == 15
    assert r.truncated is False
    assert r.attempts[-1].endswith(":ok")


def test_reasoning_content_is_never_returned(monkeypatch):
    """思维链绝不能出现在给客户的回复里 —— 里面有模型的自我质疑与试错。"""
    install(monkeypatch, ok("收到", include_reasoning=True))
    r = run(chat(MSGS))
    assert r.text == "收到"
    assert "我不确定" not in r.text
    assert "试试" not in r.text
    for leak in ("reasoning", "不确"):
        assert leak not in r.text


def test_payload_carries_model_and_non_stream(monkeypatch):
    calls = install(monkeypatch, ok())
    run(chat(MSGS, temperature=0.7, max_tokens=123))
    body = json.loads(calls[0].content)
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["max_tokens"] == 123
    assert abs(body["temperature"] - 0.7) < 1e-6
    assert calls[0].url.path.endswith("/chat/completions")


def test_bearer_header_is_attached(monkeypatch):
    calls = install(monkeypatch, ok())
    run(chat(MSGS))
    assert calls[0].headers["authorization"] == "Bearer fake-key"


def test_connection_pool_is_reused(monkeypatch):
    install(monkeypatch, ok())
    run(chat(MSGS))
    run(chat(MSGS))
    assert len(llm._CLIENTS) == 1, "同一 loop 内应复用同一个 client"


def test_client_gets_replaced_when_loop_changes(monkeypatch):
    """换个 loop 就必须换 client：旧 loop 那个已经关了，复用会报 Event loop is closed。

    这里刻意不用 id(loop) 做缓存 key —— asyncio.run 结束后 loop 对象被回收，
    id 会被复用。测试正是要锁死「按对象身份判定」这个约束。
    """
    install(monkeypatch, ok())
    run(chat(MSGS))
    _, first = next(iter(llm._CLIENTS.values()))
    run(chat(MSGS))  # 新的事件循环
    _, second = next(iter(llm._CLIENTS.values()))
    assert second is not first
    assert len(llm._CLIENTS) == 1, "每对 (url, model) 只保留最近那个循环的 client"


# --------------------------------------------------------------------------- #
# 配置错误：不重试、立即失败、透出上游原文
# --------------------------------------------------------------------------- #
def test_model_not_found_503_is_fatal_and_not_retried(monkeypatch):
    """核心用例：网关把「模型名不存在」报成 503。

    只看状态码会被当成服务抖动重试；真正的病因在 body 里。
    必须：只打一次 + 抛配置错 + detail 带上上游原文。
    """
    body = {"error": {"code": "model_not_found",
                      "message": "分组 deepseek-sale 下模型 xxx 无可用渠道",
                      "type": "packy_api_error"}}
    calls = install(monkeypatch, err(503, body))
    with pytest.raises(LLMConfigError) as exc:
        run(chat(MSGS))
    assert len(calls) == 1, f"配置类错误不应重试，实际打了 {len(calls)} 次"
    assert "model_not_found" in exc.value.detail
    assert "无可用渠道" in exc.value.detail


def test_model_not_found_is_fatal_even_when_wrapped_in_500(monkeypatch):
    """换另一个网关把同样的错误报成 500，判定也必须成立（看 body 不看状态码）。"""
    calls = install(monkeypatch, err(500, {"error": {"code": "model_not_found", "message": "boom"}}))
    with pytest.raises(LLMConfigError):
        run(chat(MSGS))
    assert len(calls) == 1


def test_invalid_api_key_is_fatal(monkeypatch):
    calls = install(monkeypatch, err(401, {"error": {"code": "invalid_api_key", "message": "Key 无效"}}))
    with pytest.raises(LLMConfigError) as exc:
        run(chat(MSGS))
    assert len(calls) == 1
    assert "Key 无效" in exc.value.detail


def test_quota_exhausted_is_fatal(monkeypatch):
    calls = install(monkeypatch, err(429, {"error": {"code": "insufficient_quota", "message": "余额不足"}}))
    with pytest.raises(LLMConfigError) as exc:
        run(chat(MSGS))
    assert len(calls) == 1
    assert "余额不足" in exc.value.detail


def test_chinese_plain_text_error_body_is_recognized(monkeypatch):
    """body 不是 JSON 时也要能命中中文关键词，不能因为解析失败就退化成「服务抖动」。"""
    calls = install(monkeypatch, err(503, "上游错误：分组下无可用渠道，请切换模型"))
    with pytest.raises(LLMConfigError):
        run(chat(MSGS))
    assert len(calls) == 1


def test_unknown_4xx_is_fatal(monkeypatch):
    """认不出来的 4xx 也按客户端错误处理 —— 重试不会有不同结果。"""
    calls = install(monkeypatch, err(422, {"error": {"message": "参数不对"}}))
    with pytest.raises(LLMConfigError):
        run(chat(MSGS))
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# 抖动：要重试
# --------------------------------------------------------------------------- #
def test_plain_503_without_fatal_body_is_retried(monkeypatch):
    calls = install(monkeypatch, err(503, {"error": {"message": "upstream temporarily unavailable"}}))
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))
    assert len(calls) == 3, f"可重试错误应重试到上限，实际 {len(calls)} 次"


def test_recovers_after_transient_failure(monkeypatch):
    def flaky(request: httpx.Request, n: int) -> httpx.Response:
        return ok("好了")(request, n) if n >= 2 else err(500, {"error": {"message": "boom"}})(request, n)

    install(monkeypatch, flaky)
    r = run(chat(MSGS))
    assert r.text == "好了"
    assert r.attempts[0].endswith(":retry")


def test_breaker_stops_calling_upstream(monkeypatch):
    """连续失败到阈值后必须熔断：否则每条客户消息都要白等一整轮退避。"""
    calls = install(monkeypatch, err(500, {"error": {"message": "boom"}}))
    for _ in range(3):
        with pytest.raises(LLMUnavailable):
            run(chat(MSGS))
    n_after_open = len(calls)
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))
    assert len(calls) == n_after_open, "熔断后再请求应直接失败，不再打网络"


def test_breaker_recovers_after_reset(monkeypatch):
    install(monkeypatch, err(500, {"error": {"message": "boom"}}))
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))
    llm.reset_breakers()
    install(monkeypatch, ok("恢复"))
    assert run(chat(MSGS)).text == "恢复"


# --------------------------------------------------------------------------- #
# 响应结构异常与截断
# --------------------------------------------------------------------------- #
def test_length_finish_reason_marks_truncated(monkeypatch):
    """推理模型撞到 max_tokens 会给出半句话，不能当正常答案发走。"""
    install(monkeypatch, ok("户外电源保修期是整机 24", finish="length"))
    r = run(chat(MSGS))
    assert r.truncated is True
    assert r.text.startswith("户外电源保修期是整机 24")


def test_normal_finish_is_not_truncated(monkeypatch):
    install(monkeypatch, ok())
    assert run(chat(MSGS)).truncated is False


def test_empty_content_raises(monkeypatch):
    install(monkeypatch, ok("   "))
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))


def test_empty_choices_raises(monkeypatch):
    def handler(request: httpx.Request, _n: int) -> httpx.Response:
        return httpx.Response(200, json={"choices": [], "usage": {}}, request=request)

    install(monkeypatch, handler)
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))


# --------------------------------------------------------------------------- #
# 未配置 / 自检
# --------------------------------------------------------------------------- #
def test_stub_provider_returns_stub_text(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub", raising=False)
    r = run(chat(MSGS))
    assert "模型未配置" in r.text
    assert r.endpoint == "stub" and r.degraded is True


def test_no_endpoint_raises(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "", raising=False)
    monkeypatch.setattr(settings, "llm_api_key", "", raising=False)
    monkeypatch.setattr(settings, "llm_model", "", raising=False)
    with pytest.raises(LLMUnavailable):
        run(chat(MSGS))


def test_healthcheck_ok(monkeypatch):
    install(monkeypatch, ok("pong"))
    info = run(healthcheck())
    assert info["ok"] is True
    assert info["model"] == MODEL
    assert info["replied_model"] == MODEL
    assert info["latency_ms"] >= 0


def test_healthcheck_reports_upstream_detail_on_config_error(monkeypatch):
    install(monkeypatch, err(503, {"error": {"code": "model_not_found", "message": "无可用渠道"}}))
    info = run(healthcheck())
    assert info["ok"] is False
    assert info["reason"] == "config"
    assert "无可用渠道" in info["detail"]


def test_healthcheck_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub", raising=False)
    info = run(healthcheck())
    assert info["ok"] is False and info["reason"] == "unconfigured"
    assert "stub" in info["detail"]


# --------------------------------------------------------------------------- #
# 推理预算（reasoning_effort）
#
# 实测教训：推理模型在"资料没命中 / 问题开放"时会展开上万字思维链，同一句开放性提问
# 默认 54s（思维链 14616 字），压到 minimal 后 2.8s（思维链 0 字），快 19 倍。
# 客服是实时对话，这个参数一旦丢失，慢的问题会悄悄回来。
# --------------------------------------------------------------------------- #
def test_reasoning_effort_sent_when_configured(monkeypatch):
    """配置了推理预算就必须发给上游，否则思维链会把回复拖到几十秒。"""
    monkeypatch.setattr(settings, "llm_reasoning_effort", "minimal", raising=False)
    calls = install(monkeypatch, ok("你好"))
    run(chat([{"role": "user", "content": "hi"}]))

    payload = json.loads(calls[0].content)
    assert payload.get("reasoning_effort") == "minimal", payload


def test_reasoning_effort_omitted_when_blank(monkeypatch):
    """置空时不传该字段——非推理模型不认这个参数，硬传可能被上游拒。"""
    monkeypatch.setattr(settings, "llm_reasoning_effort", "", raising=False)
    calls = install(monkeypatch, ok("你好"))
    run(chat([{"role": "user", "content": "hi"}]))

    payload = json.loads(calls[0].content)
    assert "reasoning_effort" not in payload, payload


# --------------------------------------------------------------------------- #
# 思考开关（thinking）
#
# DeepSeek 官方 API 的推理模型默认开思考，实测同一句开放性提问：
#   thinking.enabled  → 7s，且思维链会把 max_tokens 吃光，正文返回 0 字（空答）
#   thinking.disabled → 1.0s，思维 0 字，正文正常
# 客服是实时对话，这个开关一旦丢失，慢和空答会一起回来。
# --------------------------------------------------------------------------- #
def test_thinking_sent_when_configured(monkeypatch):
    """配置了思考开关就必须发给上游。"""
    monkeypatch.setattr(settings, "llm_thinking", "disabled", raising=False)
    calls = install(monkeypatch, ok("你好"))
    run(chat([{"role": "user", "content": "hi"}]))

    payload = json.loads(calls[0].content)
    assert payload.get("thinking") == {"type": "disabled"}, payload


def test_thinking_omitted_when_blank(monkeypatch):
    """置空时不传该字段——不认识这个参数的上游可能直接拒请求。"""
    monkeypatch.setattr(settings, "llm_thinking", "", raising=False)
    calls = install(monkeypatch, ok("你好"))
    run(chat([{"role": "user", "content": "hi"}]))

    payload = json.loads(calls[0].content)
    assert "thinking" not in payload, payload
