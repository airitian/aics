"""重排模型（Rerank）适配层：二阶段检索的第二段。

为什么需要它
------------
向量检索是**单向量粗排**：query 和文档各自压成一个向量再比相似度，
语义细节在这个压缩里丢掉了。实测（verify_retrieval.py，Amazon 手册 75 题）：

    bge-m3 单独              R@1=69.3%  R@5=86.7%
    bge-m3 + Qwen3-Reranker  R@1=82.7%  R@5=93.3%

R@1 提升 13.4 个百分点 —— 而换更强的 embedding 模型（Qwen3-Embedding-8B）
只有 2.6pp，且差距落在噪声内。**重排是本项目检索质量最大的单一杠杆。**

rerank 是交叉编码器：query 与候选**拼在一起**过模型，能看见词与词的对应关系，
代价是必须逐对计算，所以只能排在粗排之后、候选数受限（本项目粗排 20 条）。

设计要点
--------
1. **失败必须优雅降级**：rerank 是增强项，挂了应该退回向量排序，
   绝不能让一轮客户会话因为没有重排就答不上来。所以对外只有一个软失败入口
   `rerank()`，返回 None 表示「不可用，请用原顺序」。
2. **错误判定先读 body**：中转网关可能把「模型名写错」报成 5xx，
   只按状态码重试会把配置错误伪装成服务抖动，排查方向整个跑偏。
3. **连接池按事件循环对象缓存**（不用 id()），理由见 embedding.py 的同名注释。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

import httpx

from app.config import settings
from app.resilience import Breaker, is_retryable_status

logger = logging.getLogger("aics.rerank")

_CLIENTS: dict[tuple[str, str], tuple[object, httpx.AsyncClient]] = {}
_BREAKERS: dict[tuple[str, str], Breaker] = {}
_LOCK = threading.Lock()

# 上游错误体里出现这些词，说明是**配置写错了**（模型名/Key/额度），
# 重试一万次也没用，必须直接把原话透出，让运维一眼看到。
_FATAL_HINTS = (
    "model_not_found",
    "invalid_api_key",
    "insufficient_quota",
    "无可用渠道",
    "余额不足",
    "拼写",
    "有误",
    "不存在",
)


class RerankUnavailable(Exception):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def enabled() -> bool:
    """是否启用重排。未配置模型或显式关闭时走纯向量排序。"""
    return (
        settings.rerank_provider == "api"
        and bool(settings.rerank_model)
        and bool(settings.rerank_base_url)
    )


def _key() -> tuple[str, str]:
    return (settings.rerank_base_url.rstrip("/"), settings.rerank_model)


def _running_loop() -> object | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _client(key: tuple[str, str]) -> httpx.AsyncClient:
    """复用连接池；必须按事件循环对象分别缓存（不能用 id()，详情见 embedding.py）。"""
    loop = _running_loop()
    with _LOCK:
        hit = _CLIENTS.get(key)
        if hit is not None and hit[0] is not loop:
            _CLIENTS.pop(key, None)
            hit = None
        if hit is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.rerank_timeout),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            )
            _CLIENTS[key] = (loop, client)
            return client
        return hit[1]


def _breaker(key: tuple[str, str]) -> Breaker:
    with _LOCK:
        b = _BREAKERS.get(key)
        if b is None:
            b = Breaker(
                threshold=settings.rerank_breaker_threshold,
                cooldown=settings.rerank_breaker_cooldown,
            )
            _BREAKERS[key] = b
        return b


async def aclose() -> None:
    with _LOCK:
        clients = [c for _, c in _CLIENTS.values()]
        _CLIENTS.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:  # pragma: no cover - 关闭失败不影响退出
            pass


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except Exception:
        return (resp.text or "").strip()[:300]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code")
            if msg:
                return str(msg)[:300]
        if isinstance(err, str):
            return err[:300]
        detail = body.get("detail") or body.get("message")
        if detail:
            return str(detail)[:300]
        return json.dumps(body, ensure_ascii=False)[:300]
    return str(body)[:300]


def _looks_fatal(detail: str) -> bool:
    return any(h in detail for h in _FATAL_HINTS)


async def rerank_scores(query: str, documents: list[str]) -> list[float]:
    """对候选打相关性分。顺序与 documents 一致；失败抛 RerankUnavailable。"""
    key = _key()
    breaker = _breaker(key)
    if breaker.is_open():
        raise RerankUnavailable(
            "重排模型暂时不可用（熔断冷却中）", detail="circuit breaker open"
        )

    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
    }
    if settings.rerank_top_n > 0:
        payload["top_n"] = min(settings.rerank_top_n, len(documents))

    attempts = max(1, settings.rerank_retries)
    delay = max(0.0, settings.rerank_retry_delay)
    last: BaseException | None = None

    for i in range(attempts):
        try:
            resp = await _client(key).post(f"{key[0]}/rerank", json=payload)
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"upstream {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise RerankUnavailable(
                    "重排模型返回了非 JSON 响应，请检查 RERANK_BASE_URL 是否指向正确服务",
                    detail=(resp.text or "")[:200],
                ) from exc

            breaker.on_success()
            results = data.get("results") or []
            # 上游不保证顺序，必须按 index 归位，否则分数会错配到别的候选上
            results = sorted(results, key=lambda x: x.get("index", 0))
            scores = [float(r.get("relevance_score") or 0.0) for r in results]
            if len(scores) != len(documents):
                raise RerankUnavailable(
                    f"重排返回条数({len(scores)})与候选数({len(documents)})不符，已放弃重排",
                    detail="",
                )
            return scores

        except RerankUnavailable:
            # 响应体层面的问题（非 JSON / 条数不符）不该计入熔断，也不必重试
            raise
        except httpx.HTTPStatusError as exc:
            detail = _error_detail(exc.response)
            last = exc
            if not is_retryable_status(exc.response.status_code):
                breaker.on_failure()
                raise RerankUnavailable(
                    f"重排模型请求被拒绝（HTTP {exc.response.status_code}）",
                    detail=detail,
                ) from exc
            # 5xx 也要先看 body：网关常把「模型名不存在」报成 503
            if _looks_fatal(detail):
                raise RerankUnavailable(
                    f"重排模型配置有误：{detail}",
                    detail=detail,
                ) from exc
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            last = exc

        if i < attempts - 1:
            logger.warning("重排模型第 %d/%d 次调用失败（将重试）：%s", i + 1, attempts, last)
            await asyncio.sleep(delay * (2**i))

    breaker.on_failure()
    detail = f"{type(last).__name__}: {last}" if last else "unknown"
    raise RerankUnavailable("重排模型调用失败，已降级为向量排序", detail=detail)


async def rerank(query: str, documents: list[str]) -> list[float] | None:
    """软失败入口：拿不到分数就返回 None，调用方按原顺序继续。

    重排是增强项而不是必需品 —— 它挂了不该让客户这一轮拿不到答案。
    """
    if not enabled() or not documents:
        return None
    try:
        return await rerank_scores(query, documents)
    except RerankUnavailable as exc:
        logger.warning("重排不可用，已降级为向量排序：%s | %s", exc, exc.detail)
        return None
    except Exception as exc:  # pragma: no cover - 兜底，避免异常冒泡打断检索
        logger.warning("重排异常，已降级为向量排序：%s", exc)
        return None


async def healthcheck() -> dict:
    if not enabled():
        return {"ok": True, "provider": "off", "detail": "未启用重排（RERANK_PROVIDER != api）"}
    try:
        scores = await rerank_scores("测试", ["测试文档一", "测试文档二"])
        return {
            "ok": True,
            "provider": "api",
            "model": settings.rerank_model,
            "detail": f"连通，返回 {len(scores)} 条分数",
        }
    except RerankUnavailable as exc:
        return {"ok": False, "provider": "api", "model": settings.rerank_model, "detail": f"{exc} | {exc.detail}"}
