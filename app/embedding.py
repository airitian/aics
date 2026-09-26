"""向量模型适配层（检索召回）。

EMBED_PROVIDER=api   → 调 OpenAI 兼容 /embeddings 接口（生产）。
                       本项目接入 **Gitee AI 模力方舟** 的 `bge-m3`：
                       POST https://ai.gitee.com/api/v1/embeddings
                       Authorization: Bearer <EMBED_API_KEY>
                       输出 **1024 维**，支持一次传多条 input。
EMBED_PROVIDER=local → 本地确定性哈希向量，**仅供离线开发/单测跑通链路**，
                       不具备真实语义能力，生产务必切到 api。

三个工程要点（都是上线后才暴露的坑）：

1. **连接池复用**：入库会按 `EMBED_BATCH` 分批，若每批都 new 一个 AsyncClient，
   等于每批都重做一次 TLS 握手 —— 公网 RTT 直接翻倍。这里用模块级客户端复用。
2. **维度守卫**：向量库集合的维度一经创建即固定。这里把「模型**实际返回**的维度」
   与 `EMBED_DIM` 对齐校验，配错时当场给出可执行修法，而不是等 Qdrant 抛
   dimension mismatch 这种隔了一层的错误（排查时很难联想到是模型配错了）。
3. **抖动重试 + 真故障熔断**：与向量库同一套策略，见 `app/resilience.py`。
   另外区分「可重试」与「请求写错了」：4xx/鉴权失败既不该重试也不该计入熔断，
   否则会把一个常量配置 bug 伪装成「模型服务不稳定」，排查方向被彻底带偏。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
from typing import Any
from dataclasses import dataclass, field

import httpx

from app.config import settings
from app.resilience import Breaker, is_retryable_status

logger = logging.getLogger("aics.embedding")

_CJK = r"\u4e00-\u9fff"


class EmbeddingUnavailable(Exception):
    """向量模型不可用（网络/上游故障/鉴权错误）。

    单独定义类型，是为了让上层能把「模型调不通」和「代码 bug」分开：
    前者映射成 503（可恢复），并让对话走降级话术 + 转人工，而不是编造答案。
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


class EmbeddingDimensionMismatch(EmbeddingUnavailable):
    """模型实际输出维度 != EMBED_DIM。

    继承 EmbeddingUnavailable 让上层统一按「向量层不可用」处理（503 / 降级 + 转人工），
    但 message 必须给出可执行修法 —— 只报一句「模型不可用」会让人去查网络，
    而真正的问题是维度配错了。
    """

    def __init__(self, got: int, expect: int):
        super().__init__(
            f"向量模型实际输出 {got} 维，与配置的 EMBED_DIM={expect} 不一致，"
            f"写入向量库会导致维度冲突，已中止。请改 EMBED_DIM={got}（或换成 {expect} 维的模型）；"
            f"若向量库集合已按 {expect} 维建过，还必须删掉集合并重新入库。",
            detail=(
                "bge-m3 的输出维度是 1024。若只是配置漏改，把 EMBED_DIM 改成模型真实维度即可；"
                "若确实换了模型，旧集合的向量不可复用，必须重建集合 + 重新上传文档。"
            ),
        )
        self.got = got
        self.expect = expect


@dataclass
class EmbedResult:
    vectors: list[list[float]] = field(default_factory=list)
    dim: int = 0
    tokens: int = 0
    provider: str = "api"


# --------------------------------------------------------------------------- #
# 客户端复用 + 熔断
# --------------------------------------------------------------------------- #
# (base_url, model) -> AsyncClient
# (base_url, model) -> (创建它的 event loop, 客户端)。每对 (url, model) 只保留最近那个循环的。
_CLIENTS: dict[tuple[str, str], tuple[Any, httpx.AsyncClient]] = {}
# (base_url, model) -> 熔断器
_BREAKERS: dict[tuple[str, str], Breaker] = {}
_LOCK = threading.Lock()


def _key() -> tuple[str, str]:
    return (settings.embed_base_url.rstrip("/"), settings.embed_model)


def _running_loop() -> Any:
    """当前事件循环对象；取不到（理论上不会）时返回 None。

    **一定要用对象本身，不能用 id()**：asyncio.run 结束后循环对象会被回收，
    id 是会被复用的 —— 下一个 loop 可能拿到和上一个一模一样的 id。
    一旦按 id 判定，就会命中上一个（已关闭）循环留下的客户端，
    症状正是 `Event loop is closed`，而且是偶发的：取决于 GC 是否刚好复用了那个地址。
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _client(key: tuple[str, str]) -> httpx.AsyncClient:
    """复用连接池：分批入库时不必每批重做 TLS 握手。

    **必须按事件循环分别缓存**：httpx 的 AsyncClient 内部持有的连接/锁是绑在
    创建它的那个 event loop 上的。缓存不区分循环，跨循环复用会直接抛
    `Event loop is closed` —— 而且是在**非请求路径**上静默失败
    （bootstrap 里每个演示租户各跑一次 asyncio.run，第二个租户的知识库
    就入库失败了，日志只有一句 warning，表现为「演示数据查不到」）。
    """
    loop = _running_loop()
    with _LOCK:
        # 若缓存里的客户端属于**别的**循环对象，说明那个循环已经结束了，
        # 直接丢弃（留着也没法用）。进程内循环数量很少（bootstrap 一次 + 服务一次），
        # 所以最多残留一个，不会无限增长。
        hit = _CLIENTS.get(key)
        if hit is not None and hit[0] is not loop:
            _CLIENTS.pop(key, None)
            hit = None
        if hit is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.embed_timeout),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                headers={"Authorization": f"Bearer {settings.embed_api_key}"},
            )
            _CLIENTS[key] = (loop, client)
            return client
        return hit[1]


def _breaker(key: tuple[str, str]) -> Breaker:
    with _LOCK:
        b = _BREAKERS.get(key)
        if b is None:
            b = Breaker(
                threshold=settings.embed_breaker_threshold,
                cooldown=settings.embed_breaker_cooldown,
            )
            _BREAKERS[key] = b
        return b


async def aclose() -> None:
    """应用关闭时释放连接池（否则 uvicorn 退出会有未关闭连接的告警）。"""
    with _LOCK:
        clients = [c for _, c in _CLIENTS.values()]
        _CLIENTS.clear()
    for client in clients:
        try:
            await client.aclose()
        except Exception:  # pragma: no cover - 关闭失败不影响退出
            pass


# --------------------------------------------------------------------------- #
# 本地开发用：确定性哈希向量
# --------------------------------------------------------------------------- #
_word_re = re.compile(rf"[{_CJK}]|[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    raw = _word_re.findall((text or "").lower())
    out: list[str] = []
    # 相邻 CJK 单字组成二元组，提升中文区分度
    for i, tok in enumerate(raw):
        out.append(tok)
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if len(nxt) == 1 and "\u4e00" <= nxt <= "\u9fff":
                out.append(tok + nxt)
    return out


def _local_embed_one(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    toks = _tokens(text)
    if not toks:
        return vec
    for tok in toks:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# --------------------------------------------------------------------------- #
# 远程接口
# --------------------------------------------------------------------------- #
def _error_detail(resp: httpx.Response) -> str:
    """把上游错误体压成一行可读文案。

    Gitee AI 的错误形如 {"error":{"code":"400","message":"模型名称…有误…"}}，
    把 message 原样带出来，排查时能直接看到「模型名写错了」而不是一个裸 400。
    """
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


def _check_dim(got: int, expect: int) -> None:
    if not got or expect <= 0 or got == expect:
        return
    raise EmbeddingDimensionMismatch(got, expect)


async def _post_once(payload: dict) -> dict:
    key = _key()
    resp = await _client(key).post(f"{key[0]}/embeddings", json=payload)
    if resp.status_code >= 400:
        # 抛出带状态码的异常，交给调用方按「可重试 / 写错了」分流
        raise httpx.HTTPStatusError(
            f"upstream {resp.status_code}", request=resp.request, response=resp
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise EmbeddingUnavailable(
            "向量模型返回了非 JSON 响应，请检查 EMBED_BASE_URL 是否指向了正确的服务",
            detail=(resp.text or "")[:200],
        ) from exc


async def _embed_api(batch: list[str]) -> tuple[list[list[float]], int]:
    key = _key()
    breaker = _breaker(key)
    if breaker.is_open():
        raise EmbeddingUnavailable(
            "向量模型暂时不可用，请稍后重试",
            detail="circuit breaker open（连续失败后进入冷却，已跳过网络请求）",
        )

    payload = {"model": settings.embed_model, "input": batch}
    attempts = max(1, settings.embed_retries)
    delay = max(0.0, settings.embed_retry_delay)
    last: BaseException | None = None

    for i in range(attempts):
        try:
            data = await _post_once(payload)
            breaker.on_success()
            items = data.get("data") or []
            # 上游不保证按顺序返回，必须按 index 归位，否则向量与文本会错配
            items = sorted(items, key=lambda x: x.get("index", 0))
            vectors = [list(map(float, it.get("embedding") or [])) for it in items]
            if vectors:
                _check_dim(len(vectors[0]), settings.embed_dim)
            usage = data.get("usage") or {}
            tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
            return vectors, tokens
        except EmbeddingDimensionMismatch:
            # 配置问题：重试和熔断都救不了，直接冒泡（且不能计入熔断）
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last = exc
            if not is_retryable_status(status):
                # 4xx（除 429）= 请求本身写错了：模型名/Key/参数不对。
                # 重试一万次也没用，而且把它算进熔断会把「配置 bug」说成「服务不稳定」。
                raise EmbeddingUnavailable(
                    f"向量模型请求被拒绝（HTTP {status}），通常是模型名或 API Key 配置有误",
                    detail=_error_detail(exc.response),
                ) from exc
            if i == attempts - 1:
                break
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            last = exc
            if i == attempts - 1:
                break
        logger.warning(
            "向量模型第 %d/%d 次调用失败（将重试）：%s", i + 1, attempts, last
        )
        await asyncio.sleep(delay * (2 ** i))

    breaker.on_failure()
    detail = f"{type(last).__name__}: {last}" if last else "unknown"
    if isinstance(last, httpx.HTTPStatusError):
        detail = f"HTTP {last.response.status_code}: {_error_detail(last.response)}"
    raise EmbeddingUnavailable("向量模型暂时不可用，请稍后重试", detail=detail) from last


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
async def embed_texts(texts: list[str]) -> EmbedResult:
    if not texts:
        return EmbedResult(vectors=[], dim=settings.embed_dim, provider=settings.embed_provider)

    if settings.embed_provider == "local":
        vectors = [_local_embed_one(t, settings.embed_dim) for t in texts]
        return EmbedResult(
            vectors=vectors,
            dim=settings.embed_dim,
            tokens=sum(len(t or "") for t in texts) // 2,
            provider="local",
        )

    if not (settings.embed_base_url and settings.embed_api_key and settings.embed_model):
        raise EmbeddingUnavailable(
            "未配置向量模型端点（EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL）"
        )

    all_vectors: list[list[float]] = []
    total_tokens = 0
    size = max(1, settings.embed_batch)
    for i in range(0, len(texts), size):
        # 空串会让部分上游报错；本地兜底成空格（bge-m3 返回真实向量，不影响语义）
        batch = [t if (t or "").strip() else " " for t in texts[i : i + size]]
        vectors, tokens = await _embed_api(batch)
        if len(vectors) != len(batch):
            raise EmbeddingUnavailable(
                f"向量模型返回条数与请求不一致（{len(vectors)}/{len(batch)}）"
            )
        all_vectors.extend(vectors)
        total_tokens += tokens

    dim = len(all_vectors[0]) if all_vectors else settings.embed_dim
    _check_dim(dim, settings.embed_dim)
    return EmbedResult(vectors=all_vectors, dim=dim, tokens=total_tokens, provider="api")


async def embed_query(text: str) -> tuple[list[float], int]:
    res = await embed_texts([text])
    if not res.vectors:
        raise EmbeddingUnavailable("检索向量生成为空")
    return res.vectors[0], res.tokens


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
async def healthcheck() -> dict:
    """启动自检用：真打一次接口，确认「能通 + 维度对」。

    为什么不只检查配置是否填了：填了但填错（Key 失效、模型名下架、维度不符）
    是最常见的上线事故，而这几种错在**第一条客户消息**才会暴露成 503。
    这里花一次几十毫秒的调用，把它提前到启动日志里。
    """
    base = {
        "provider": settings.embed_provider,
        "model": settings.embed_model or None,
        "base_url": settings.embed_base_url or None,
        "expected_dim": settings.embed_dim,
    }
    if settings.embed_provider == "local":
        return {
            **base,
            "ok": True,
            "dim": settings.embed_dim,
            "note": "本地占位向量（无真实语义能力），生产请设 EMBED_PROVIDER=api",
        }
    try:
        res = await embed_texts(["健康检查"])
    except EmbeddingUnavailable as exc:
        return {**base, "ok": False, "error": exc.message, "detail": exc.detail}
    return {**base, "ok": True, "dim": res.dim, "probe_tokens": res.tokens}
