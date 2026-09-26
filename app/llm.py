"""对话模型适配层（OpenAI 兼容接口）。

设计要点（对应上线准备清单第 2 条）：
- 支持主端点 + 备用端点，主端点失败自动切换，避免「模型一挂全平台租户都发布不了」；
- 全部端点失败时抛 LLMUnavailable，调用方必须明确告知用户，**严禁编造答案**；
- LLM_PROVIDER=stub 只用于本地跑通链路，会明确回复「模型未配置」，不产生假答案。

两个必须在这里说清的坑（都是实测出来的，不是理论）：

**坑 1：网关会把「写错了」报成 503。**
实测 PackyAPI：模型名下架/写错 → 返回 **503**（不是 404/400），body 里才是
真正的 `model_not_found`。按状态码它属于「服务端错误」，一股脑重试的下场是：
白等两轮退避、照样失败，最后报「对话模型暂时不可用，请稍后重试」——
把**配置写错了**伪装成**服务在抖动**，排查方向整个跑偏。
所以这里不能只看状态码，必须读 body 判断是不是「客户端性质」的错误。

**坑 2：推理模型的思维链和正式回答在同一个 message 里返回。**
实测 deepseek-v4-flash：一次问答用了 22 个 completion tokens，其中 **19 个是
reasoning_tokens**（思考消耗），正式回答只有几个字。
两个后果：
  a) `reasoning_content` 是模型的思考过程，**绝不能发给客户**——它包含
     「我不确定，要不要编一个」这类自言自语，发出去就是自曝。这里只读 `content`。
  b) 思考 token **吃的是 max_tokens 同一份预算**，且**照常计费**。
     配得太小会撞上限 → `finish_reason="length"` → 客户收到半句话。
     这种截断必须被识别出来（`truncated`），不能当正常答案静默发走。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.resilience import Breaker, is_retryable_status

logger = logging.getLogger("aics.llm")

# 命中即判定为「客户端性质」的错误：重试一百次也是同样的结果，只会浪费时间和额度
_FATAL_HINTS = (
    "model_not_found",
    "model_no_permission",
    "invalid_api_key",
    "invalid_token",
    "authentication_error",
    "permission_denied",
    "insufficient_quota",
    "quota_exceeded",
    "insufficient_balance",
    "no_available_channel",
    "无可用渠道",
    "模型不存在",
    "余额不足",
    "额度不足",
    "鉴权失败",
    "令牌",
)

# (base_url, model) -> 熔断器。真故障时冷却期内直接失败，不让每条客户消息都白等一轮退避
_BREAKERS: dict[tuple[str, str], Breaker] = {}

# (base_url, model) -> (创建它的 event loop, 客户端)
_CLIENTS: dict[tuple[str, str], tuple[Any, httpx.AsyncClient]] = {}


class LLMUnavailable(Exception):
    """所有端点均不可用。"""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


class LLMConfigError(LLMUnavailable):
    """模型配置错误（模型名不存在 / Key 无效 / 额度耗尽）。

    单独建类是为了让上层区分对待：这类错误**切备用端点也没用**，
    且必须把上游原文透出去 —— 运维第一眼就要知道是配置写错了，而不是去查网络。
    """


@dataclass
class LLMResult:
    text: str
    model: str = ""
    endpoint: str = "primary"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degraded: bool = False
    error: str = ""
    attempts: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str) -> int:
    """无 usage 返回时的粗略估算（中文约 1 字 1 token，英文约 4 字符 1 token）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def _fatal_reason(status: int, body: str) -> str | None:
    """从错误响应体里识别「重试也没用」的配置类错误。

    返回人类可读的原因；判定不了就返回 None（交给调用方按状态码正常重试）。
    """
    if not body:
        return None
    # body 可能是 JSON，也可能直接是纯文本错误页，两种都要能吃
    text = body
    code = ""
    message = ""
    try:
        import json

        data = json.loads(body)
        err = data.get("error") or {}
        if isinstance(err, dict):
            code = str(err.get("code") or "")
            message = str(err.get("message") or err.get("msg") or "")
        else:
            message = str(err)
        if message:
            text = f"{code} {message}"
    except Exception:  # noqa: BLE001 - 不是 JSON 就用原文匹配
        pass

    low = text.lower()
    for hint in _FATAL_HINTS:
        if hint.lower() in low:
            # code 和 message 都要保留：code（`model_not_found`）是给机器和搜索引擎看的，
            # message（中文原文）是给人看的。只留一个都会让人多查一轮。
            tail = " · ".join(x for x in (code, message) if x) or body[:200]
            return f"{status} · {tail}".strip()
    return None


def _upstream_error(status: int, body: str) -> str:
    """把上游错误整理成一行可读文本（优先用上游给的 message）。"""
    if not body:
        return f"上游返回 {status}"
    try:
        import json

        data = json.loads(body)
        err = data.get("error") or data.get("message")
        if isinstance(err, dict):
            return f"{status} · {err.get('message') or err.get('code') or body[:200]}"
        if err:
            return f"{status} · {err}"
    except Exception:  # noqa: BLE001
        pass
    return f"{status} · {body[:200]}"


def _endpoints(profile: str = "main") -> list[dict]:
    """按档位返回候选端点。

    - main：主端点 + 备用端点（生产/网页测试走这里）；
    - dev：开发测试专用档（旧中转 key）。三项未配齐时**回落主档**而不是报错——
      dev 档是省钱手段，不是可用性依赖，配错了不该把测试链路整个打挂。
    """
    if profile == "dev":
        if settings.llm_dev_base_url and settings.llm_dev_api_key and settings.llm_dev_model:
            return [
                {
                    "name": "dev",
                    "base_url": settings.llm_dev_base_url.rstrip("/"),
                    "api_key": settings.llm_dev_api_key,
                    "model": settings.llm_dev_model,
                }
            ]
        return _endpoints("main")

    points: list[dict] = []
    if settings.llm_base_url and settings.llm_api_key and settings.llm_model:
        points.append(
            {
                "name": "primary",
                "base_url": settings.llm_base_url.rstrip("/"),
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
            }
        )
    if settings.llm_fallback_base_url and settings.llm_fallback_api_key and settings.llm_fallback_model:
        points.append(
            {
                "name": "fallback",
                "base_url": settings.llm_fallback_base_url.rstrip("/"),
                "api_key": settings.llm_fallback_api_key,
                "model": settings.llm_fallback_model,
            }
        )
    return points


STUB_TEXT = (
    "【模型未配置】当前服务未接入对话模型（LLM_PROVIDER=stub）。"
    "请在环境变量中配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后重试。"
)


def _client(point: dict) -> httpx.AsyncClient:
    """复用连接池：每条客户消息都新建 client 会重做一次 TLS 握手，白白多花上百毫秒。

    key 里必须带上**事件循环对象本身**：httpx 的 AsyncClient 绑定创建它的 loop，
    跨 asyncio.run 复用会报 "Event loop is closed"（本项目 bootstrap 里踩过）。

    注意**不能用 id(loop) 代替 loop 对象**：asyncio.run 结束后循环对象被回收，
    而 id 是会被分配给下一个对象的。真按 id 判定，就会偶发命中上一个已关闭循环
    留下的 client —— 症状和这个 bug 本来的样子一模一样，但因为只在某些 GC 时机下
    发生，极难复现。测试 `test_client_cache_is_keyed_by_event_loop` 就是为此存在的。
    """
    try:
        loop: Any = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    key = (point["base_url"], point["model"])
    hit = _CLIENTS.get(key)
    if hit is not None and hit[0] is not loop:
        # 属于别的（多半已经结束的）循环，留着也没法用
        _CLIENTS.pop(key, None)
        hit = None
    if hit is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={"Authorization": f"Bearer {point['api_key']}"},
        )
        _CLIENTS[key] = (loop, client)
        return client
    return hit[1]


async def aclose() -> None:
    """释放所有连接池（应用关闭时调用）。"""
    while _CLIENTS:
        _, (_, client) = _CLIENTS.popitem()
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - 关闭失败不该影响退出
            pass


def _breaker(point: dict) -> Breaker:
    key = (point["base_url"], point["model"])
    br = _BREAKERS.get(key)
    if br is None:
        br = Breaker(
            threshold=max(1, settings.llm_breaker_threshold),
            cooldown=max(0.0, settings.llm_breaker_cooldown),
        )
        _BREAKERS[key] = br
    return br


def reset_breakers() -> None:
    for br in _BREAKERS.values():
        br.reset()


async def _post(point: dict, payload: dict) -> tuple[dict, str]:
    """打一次接口。成功返回 (data, "")；失败返回 (None, 原因)。

    区分三种失败：
    - fatal（`LLMConfigError` 语义）：配置写错，重试无意义 → 立即返回，不再重试
    - retryable：抖动/限流 → 抛给上层退避重试
    - 其它 4xx：也是配置/协议错 → 按 fatal 处理
    """
    resp = await _client(point).post(f"{point['base_url']}/chat/completions", json=payload)
    if resp.status_code >= 400:
        reason = _fatal_reason(resp.status_code, resp.text)
        if reason:
            raise LLMConfigError("对话模型配置有误，已停止重试", detail=reason)
        if is_retryable_status(resp.status_code):
            raise httpx.HTTPStatusError(
                _upstream_error(resp.status_code, resp.text),
                request=resp.request,
                response=resp,
            )
        raise LLMConfigError(
            "对话模型不可用（上游拒绝请求）",
            detail=_upstream_error(resp.status_code, resp.text),
        )
    return resp.json(), ""


def _parse(data: dict, point: dict) -> LLMResult:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("上游返回空 choices")

    choice = choices[0]
    message = choice.get("message") or {}
    # 只取正式回复。推理模型的 `reasoning_content` 是它的思考过程，
    # 里面有大量自我质疑和试错，绝不能出现在客户对话里。
    text = (message.get("content") or "").strip()

    finish = choice.get("finish_reason")
    truncated = finish == "length"
    if truncated:
        # 半句话发出去，客户会以为客服在敷衍。这里不能静默，要留痕。
        logger.warning(
            "回答被长度限制截断 endpoint=%s model=%s finish_reason=length max_tokens=%s",
            point["name"], point["model"], settings.llm_max_tokens,
        )

    usage = data.get("usage") or {}
    return LLMResult(
        text=text,
        model=data.get("model") or point["model"],
        endpoint=point["name"],
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        degraded=point["name"] != "primary",
        truncated=truncated,
    )


async def chat(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    profile: str = "main",
) -> LLMResult:
    if settings.llm_provider == "stub":
        prompt = "\n".join(m.get("content", "") for m in messages)
        return LLMResult(
            text=STUB_TEXT,
            model="stub",
            endpoint="stub",
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(STUB_TEXT),
            degraded=True,
            error="llm_not_configured",
        )

    points = _endpoints(profile)
    if not points:
        raise LLMUnavailable("未配置任何可用的对话模型端点，无法生成回复")

    last_error = ""
    attempts: list[str] = []
    # 第一个「配置类」错误要留到最后再抛：它比「服务不可用」有用得多。
    # 明明是模型名写错，却报一句「模型暂时不可用，请稍后重试」，运维会去查网络。
    first_config_error: LLMConfigError | None = None
    temp = settings.llm_temperature if temperature is None else temperature
    max_out = settings.llm_max_tokens if max_tokens is None else max_tokens
    # 思维链参数按档位取：dev 档默认不传（旧中转未验证透传行为），保持其历史可用配置。
    thinking = settings.llm_thinking
    effort = settings.llm_reasoning_effort
    if profile == "dev":
        thinking = settings.llm_dev_thinking
        effort = settings.llm_dev_reasoning_effort

    for point in points:
        br = _breaker(point)
        if br.is_open():
            attempts.append(f"{point['name']}#0:breaker-open")
            continue

        payload = {
            "model": point["model"],
            "messages": messages,
            "temperature": max(0.0, min(1.5, temp)),
            "max_tokens": max_out,
            "stream": False,
        }
        # 压住推理模型的思维链：客服问答不需要上万字的内部推理，
        # 那部分时间客户全在干等（见 config.llm_thinking 注释里的实测数据）。
        # 优先用 thinking 开关（能彻底关掉，且不占 max_tokens 预算）；
        # 开关没配时退而求其次用档位参数（官方只认 low/high/max）。
        if thinking:
            payload["thinking"] = {"type": thinking}
        if effort:
            payload["reasoning_effort"] = effort

        for attempt in range(1, max(1, settings.llm_max_retries) + 1):
            try:
                data, _ = await _post(point, payload)
                result = _parse(data, point)
                if not result.text:
                    raise ValueError("上游返回空内容")
                result.attempts = attempts + [f"{point['name']}#{attempt}:ok"]
                br.on_success()
                return result
            except LLMConfigError as exc:
                # 配置类错误：重试和切备用端点都救不了，记住原文交给最后抛出
                if first_config_error is None:
                    first_config_error = exc
                last_error = exc.detail or exc.message
                attempts.append(f"{point['name']}#{attempt}:fatal")
                logger.error("对话模型配置错误 endpoint=%s：%s", point["name"], last_error)
                # 熔断与否：这里是常量错误，重试必然失败，直接打开熔断器
                # 让后续客户消息立刻转人工，而不是每条都打一次上游
                br.on_failure()
                break
            except ValueError as exc:
                # 响应结构异常/空内容 —— 换端点再试，同一端点重试意义不大
                last_error = str(exc)
                attempts.append(f"{point['name']}#{attempt}:bad-response")
                br.on_failure()
                break
            except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(f"{point['name']}#{attempt}:retry")
                br.on_failure()
                if attempt < max(1, settings.llm_max_retries):
                    await asyncio.sleep(0.4 * attempt)

    logger.warning("所有模型端点均失败 attempts=%s err=%s", attempts, last_error)
    if first_config_error is not None:
        # LLMConfigError 是 LLMUnavailable 的子类，现有调用方无需改动即可继续捕获；
        # 但异常类型与 detail 必须说清是「配置错」，而不是让人去查网络。
        first_config_error.attempts = attempts
        raise first_config_error
    raise LLMUnavailable("对话模型暂时不可用，请稍后重试", detail=last_error)


async def healthcheck() -> dict:
    """真打一次接口，确认「Key + 模型名 + 端点」三者都对。

    只测连通不够：Key 失效和模型名下架都发生在 HTTP 层之外，
    不在启动时不打这一枪，就要等第一条客户消息来暴露。
    """
    if settings.llm_provider == "stub":
        return {
            "ok": False,
            "provider": "stub",
            "model": None,
            "configured": False,
            "reason": "unconfigured",
            "detail": "LLM_PROVIDER=stub，未接入对话模型，对话会回复「模型未配置」",
        }

    point = next((p for p in _endpoints() if p["name"] == "primary"), None)
    info = {
        "provider": settings.llm_provider,
        "base_url": (point or {}).get("base_url") or settings.llm_base_url or "-",
        "model": (point or {}).get("model") or settings.llm_model or "-",
        "configured": point is not None,
    }
    if point is None:
        return {**info, "ok": False, "reason": "unconfigured",
                "detail": "LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 未配置完整"}

    started = time.monotonic()
    try:
        probe = {
            "model": point["model"],
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "stream": False,
        }
        if settings.llm_thinking:
            probe["thinking"] = {"type": settings.llm_thinking}
        if settings.llm_reasoning_effort:
            probe["reasoning_effort"] = settings.llm_reasoning_effort
        data, _ = await _post(point, probe)
        result = _parse(data, point)
    except LLMConfigError as exc:
        return {**info, "ok": False, "reason": "config",
                "detail": exc.detail or exc.message,
                "latency_ms": round((time.monotonic() - started) * 1000)}
    except Exception as exc:  # noqa: BLE001 - 自检不允许把启动打挂
        return {**info, "ok": False, "reason": "unreachable",
                "detail": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.monotonic() - started) * 1000)}

    if not result.text:
        return {**info, "ok": False, "reason": "empty",
                "detail": "上游返回 200 但内容为空"}

    return {
        **info,
        "ok": True,
        "replied_model": result.model,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
