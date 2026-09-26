"""视觉模型适配层：把图片**读成文字**，让图里的信息能进知识库。

它解决什么问题
--------------
embedding 是纯文本的，图片本身进不了向量库。文档里那些写满参数的表格截图、
设备安装图、流程图，如果不处理，入库后就只剩一句占位符 —— 客户问到图里的
内容时 AI 没有任何依据，只能答「读不到」或者顺着上下文编。

这里用一次视觉模型，把图里的字抄出来，之后它就和正文一样参与检索。

它**不是以图搜图**
------------------
以图搜图要在检索端把「客户发的图」和「库里的图」放进同一个向量空间，
需要多模态 embedding，还得单独建图片索引、双路召回再融合。

这里完全不动检索链路：视觉能力只在**入库时**用一次，产出的就是普通文本。
检索、重排、回答依旧是全文本流程，向量库和 embedding 都不用换。

为什么模型报错要原话透出
------------------------
和 llm.py / rerank.py 同理由：中转网关常把「模型名写错」报成 5xx。
只按状态码重试，会把配置错误伪装成服务抖动，排查方向整个跑偏。
所以判定顺序是**先读 body**，命中关键词就不重试、直接把上游原话带上。

失败边界
--------
这是**入库阶段**的能力，不在客户等待的关键路径上。任何一张图失败都只影响
那一张图，必须降级为「保留原占位符」，绝不能让整篇文档上传失败 ——
宁可比正文少一块内容，也不能丢整个文档。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading

import httpx

from app.config import settings
from app.resilience import Breaker, is_retryable_status

logger = logging.getLogger("aics.vision")

_CLIENTS: dict[str, tuple[object, httpx.AsyncClient]] = {}
_BREAKERS: dict[str, Breaker] = {}
_LOCK = threading.Lock()

_FATAL_HINTS = (
    "model_not_found",
    "invalid_api_key",
    "insufficient_quota",
    "无可用渠道",
    "余额不足",
    "不存在",
    "拼写",
    "有误",
)


# --------------------------------------------------------------------------- #
# 提示词：决定产出能不能用，是整个模块最值钱的部分
# --------------------------------------------------------------------------- #
_SYSTEM = "你是文档数字化助手，负责把图片内容转写成可供全文检索的正文。"

_USER = """这张图来自一份文档，请把它变成可以被全文检索的正文。

按以下优先级处理：
1. 完整提取图中**所有可读文字**（标题、表头、每个单元格、标签、注释、数值和单位），
   逐条列出，不要概括、不要合并同类项。检索要靠这些细节命中。
2. 图中是表格时，用 markdown 表格还原，行列必须齐全；
   若是转置表（第一列是属性名、其余列是型号），保持它本来的方向，不要自行转置。
3. 图中是流程图/结构图/界面截图时，说明各部分的名称与前后、上下层级关系。
4. 只有图中确实没有任何可读文字时，才用一两句话说明画面的主体是什么。
5. **看不清的一律跳过，绝不推测**。型号少认一位、价格猜一个数，
   比直接读不出来危害大得多 —— 读不出来最多答不上，猜错是在骗客户。
6. 只输出正文内容本身：不要开场白，不要"这张图展示了"，不要代码块围栏。"""

_CONTEXT_TMPL = """补充上下文（图片在文档中的相邻文字，仅供你判断图在讲什么，不要把它们抄进输出）：

{context}
"""

# 整页转写（扫描版 PDF 兜底）与单图识别的分工：
# 单图提示词围着「图里有什么」转；整页提示词围着「按阅读顺序抄字」转 ——
# 说明书页面是双栏混排的，抄串了行，参数表就会张冠李戴。
_PAGE_SYSTEM = "你是文档数字化助手，负责把扫描页逐字转写成可供全文检索的正文。"

_PAGE_USER = """这是一份文档的整页扫描图，请把它转写成纯文本，供全文检索使用。

要求：
1. 按阅读顺序转写**页面上所有可读文字**：先顶部标题，再正文；双栏/多栏布局先左栏
   后右栏；表格按行转写。
2. 遇到表格时用 markdown 表格还原，行列必须齐全；转置表（第一列是属性名）保持原方向。
3. 保留条目的编号与项目符号（1. 2. 3.、·、■ 等写成序号或短横线）。
4. 页眉、页脚、页码若只有品牌名或纯数字，跳过；若含正文信息（如服务热线、网址）要保留。
5. **看不清的一律跳过，绝不推测**。型号少认一位、参数猜一个数，比读不出来危害大得多。
6. 只输出转写内容本身：不要开场白，不要"这是第几页"，不要代码块围栏。"""


class VisionUnavailable(Exception):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def enabled() -> bool:
    return (
        settings.vision_provider == "api"
        and bool(settings.vision_model)
        and bool(settings.vision_base_url)
    )


def _running_loop() -> object | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _client() -> httpx.AsyncClient:
    key = settings.vision_base_url.rstrip("/")
    loop = _running_loop()
    with _LOCK:
        hit = _CLIENTS.get(key)
        if hit is not None and hit[0] is not loop:
            _CLIENTS.pop(key, None)
            hit = None
        if hit is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.vision_timeout),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                headers={"Authorization": f"Bearer {settings.vision_api_key}"},
            )
            _CLIENTS[key] = (loop, client)
            return client
        return hit[1]


def _breaker() -> Breaker:
    key = settings.vision_base_url.rstrip("/")
    with _LOCK:
        b = _BREAKERS.get(key)
        if b is None:
            b = Breaker(
                threshold=settings.vision_breaker_threshold,
                cooldown=settings.vision_breaker_cooldown,
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
        except Exception:  # pragma: no cover
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


def _clean(text: str) -> str:
    """去掉模型爱加的围栏和客套话。"""
    text = text.replace("```markdown", "").replace("```", "").strip()
    text = re.sub(r"^(这张图|该图|图片|图中)\s*(展示|显示|呈现)?了?[：:，,]?", "", text).strip()
    return text.strip()


async def describe_image(
    data: bytes, mime: str, *, context: str = "", mode: str = "figure"
) -> str:
    """把一张图读成文字。失败抛 VisionUnavailable，**绝不返回推测内容**。

    mode="figure"：文档里的插图（默认）；mode="page"：扫描版 PDF 的整页转写，
    用专门的逐字提示词和更高的 token 上限（整页文字远比一张插图密）。
    """
    breaker = _breaker()
    if breaker.is_open():
        raise VisionUnavailable("视觉模型暂时不可用（熔断冷却中）", "circuit breaker open")
    if len(data) > settings.vision_max_bytes:
        raise VisionUnavailable(
            f"图片超过 {settings.vision_max_bytes // 1024 // 1024}MB，未送视觉模型",
            "",
        )

    is_page = mode == "page"
    system = _PAGE_SYSTEM if is_page else _SYSTEM
    max_tokens = settings.vision_page_max_tokens if is_page else settings.vision_max_tokens
    b64 = base64.b64encode(data).decode()
    prompt = _PAGE_USER if is_page else _USER
    if context and not is_page:
        prompt += "\n\n" + _CONTEXT_TMPL.format(context=context[:800])
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    payload = {
        "model": settings.vision_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    url = settings.vision_base_url.rstrip("/") + "/chat/completions"
    attempts = max(1, settings.vision_retries)
    delay = max(0.0, settings.vision_retry_delay)
    last: BaseException | None = None

    for i in range(attempts):
        try:
            resp = await _client().post(url, json=payload)
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"upstream {resp.status_code}", request=resp.request, response=resp
                )
            body = resp.json()
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            # 推理型视觉模型会额外给 reasoning_content（思维链）。
            # 它**不是**给客户的答案，混进来会让入库文本塞满自我对话。
            text = (message.get("content") or "").strip()
            if not text:
                raise VisionUnavailable("视觉模型返回了空内容", detail="empty content")
            breaker.on_success()
            return _clean(text)

        except VisionUnavailable:
            raise
        except ValueError as exc:
            raise VisionUnavailable(
                "视觉模型返回了非 JSON 响应，请检查 VISION_BASE_URL 是否指向正确服务",
                detail="",
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = _error_detail(exc.response)
            last = exc
            if not is_retryable_status(exc.response.status_code):
                breaker.on_failure()
                raise VisionUnavailable(
                    f"视觉模型请求被拒绝（HTTP {exc.response.status_code}）", detail=detail
                ) from exc
            if any(h in detail for h in _FATAL_HINTS):
                raise VisionUnavailable(f"视觉模型配置有误：{detail}", detail=detail) from exc
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            last = exc

        if i < attempts - 1:
            logger.warning("视觉模型第 %d/%d 次调用失败（将重试）：%s", i + 1, attempts, last)
            await asyncio.sleep(delay * (2**i))

    breaker.on_failure()
    detail = f"{type(last).__name__}: {last}" if last else "unknown"
    raise VisionUnavailable("视觉模型调用失败", detail=detail)


async def describe(
    data: bytes, mime: str, *, context: str = "", mode: str = "figure"
) -> str | None:
    """软失败入口：读不出来返回 None，调用方保留原占位符。"""
    if not enabled():
        return None
    try:
        text = await describe_image(data, mime, context=context, mode=mode)
        return text or None
    except VisionUnavailable as exc:
        logger.warning("图片识别失败，已保留占位符：%s | %s", exc, exc.detail)
        return None
    except Exception as exc:  # pragma: no cover - 兜底，一张图不该拖垮整篇文档
        logger.warning("图片识别异常，已保留占位符：%s", exc)
        return None


async def describe_many(
    items: list[tuple[int, bytes, str, str]], *, mode: str = "figure"
) -> dict[int, str]:
    """批量把图读成文字。

    items: [(seq, 图片字节, mime, 上下文)]，返回 {seq: 文本}。
    mode="page" 用于扫描版 PDF 整页转写（seq 即页码）。
    失败的图不出现在返回值里 —— 调用方据此保留原占位符。
    错误细节只进日志不进返回值：调用方只需要「哪些图有了文字」，
    把错误结构一起返回曾导致元组被当成文本写库（真实事故）。

    并发上限存在的理由：视觉模型慢且贵，一篇手册几十张图全并发会瞬间打满配额，
    触发上游限流后反而一张都读不出来。
    """
    if not enabled() or not items:
        return {}

    sem = asyncio.Semaphore(max(1, settings.vision_concurrency))

    async def one(seq: int, data: bytes, mime: str, ctx: str) -> tuple[int, str | None]:
        async with sem:
            try:
                return seq, await describe_image(data, mime, context=ctx, mode=mode)
            except VisionUnavailable as exc:
                logger.warning("第 %d 张图识别失败：%s | %s", seq, exc, exc.detail)
                return seq, None
            except Exception as exc:  # pragma: no cover
                return seq, None

    results = await asyncio.gather(*(one(*it) for it in items))
    return {seq: text for seq, text in results if text}


async def healthcheck() -> dict:
    if not enabled():
        return {"ok": True, "provider": "off", "detail": "未启用图片识别（VISION_PROVIDER != api）"}
    try:
        # 12x12 纯色 PNG 探针。不能用 1x1：部分视觉模型要求边长 >10px，
        # 探针太小会被当成「模型配置错误」误报（HTTP 400 InvalidParameter）。
        import struct
        import zlib

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", 12, 12, 8, 6, 0, 0, 0)
        raw = b"".join(b"\x00" + b"\xf5\xf5\xf5\xff" * 12 for _ in range(12))
        png = (
            b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
        )
        text = await describe_image(png, "image/png")
        return {
            "ok": True,
            "provider": "api",
            "model": settings.vision_model,
            "detail": f"连通，返回 {len(text)} 字",
        }
    except VisionUnavailable as exc:
        return {"ok": False, "provider": "api", "model": settings.vision_model,
                "detail": f"{exc} | {exc.detail}"}
