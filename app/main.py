"""FastAPI 应用入口。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.bootstrap import bootstrap
from app.config import settings
from app.database import SessionLocal, init_db
from app.embedding import EmbeddingUnavailable, aclose as embed_aclose, healthcheck as embed_healthcheck
from app.llm import LLMUnavailable, aclose as llm_aclose, healthcheck as llm_healthcheck
from app.rerank import aclose as rerank_aclose, healthcheck as rerank_healthcheck
from app.ratelimit import QuotaExceeded, RateLimited
from app.routers import api_router
from app.scoping import TenantMismatch
from app.vectorstore import build_vector_store  # noqa: F401  (触发适配层导入校验)
from app.vectorstore.base import VectorStoreUnavailable

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("aics")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        # bootstrap 内部需要 asyncio.run 做演示知识库入库，放到线程里跑
        await run_in_threadpool(bootstrap, db)
    finally:
        db.close()

    # 启动自检 1：对话模型。同样真打一次 —— 「Key 失效 / 模型名写错 / 额度耗尽」
    # 这三种最常见的事故，不查就会在第一条客户消息上变成 503。
    # 注意：有些网关把「模型名不存在」也报成 503（实测 PackyAPI 如此），
    # 所以这里靠响应体判定，不能只看状态码，否则会把配置错报成服务抖动。
    llm_health = await llm_healthcheck()
    _app.state.llm_health = llm_health
    if not llm_health.get("ok"):
        logger.error(
            "对话模型自检未通过 provider=%s model=%s url=%s 原因=%s：%s\n"
            "对话会降级为「模型不可用」话术并转人工（不会编造答案）。请检查：\n"
            "  1) LLM_API_KEY 是否有效、是否被停用；\n"
            "  2) LLM_MODEL 是否为该 Key 分组下真实存在的模型名\n"
            "     （不同网关分组可用模型不同，可在 /v1/models 核对）；\n"
            "  3) LLM_BASE_URL 是否带了 /v1（OpenAI 兼容接口需要）；\n"
            "  4) 账户额度是否耗尽。",
            llm_health.get("provider"),
            llm_health.get("model") or "-",
            llm_health.get("base_url") or "-",
            llm_health.get("reason") or "-",
            llm_health.get("detail"),
        )

    # 启动自检 2：向量模型。真打一次接口 —— 「Key 失效 / 模型名下架 / 维度配错」
    # 这几种最常见的事故，如果不在这里查，就会在第一条客户消息上变成 503。
    embed_health = await embed_healthcheck()
    _app.state.embed_health = embed_health
    if not embed_health.get("ok"):
        logger.error(
            "向量模型自检未通过 provider=%s model=%s url=%s 期望维度=%s：%s\n"
            "知识库入库与检索会不可用（对话会走降级话术 + 转人工，不会编造答案）。请检查：\n"
            "  1) EMBED_API_KEY 是否有效（Gitee AI 的 Key 可在「设置 → 访问令牌」查看）；\n"
            "  2) EMBED_MODEL 是否为平台上真实存在的模型名（bge-m3）；\n"
            "  3) EMBED_BASE_URL 是否为 https://ai.gitee.com/api/v1（注意要带 /api/v1）；\n"
            "  4) EMBED_DIM 是否等于模型真实输出维度（bge-m3 = 1024）。",
            embed_health.get("provider"),
            embed_health.get("model") or "-",
            embed_health.get("base_url") or "-",
            embed_health.get("expected_dim"),
            embed_health.get("error") or embed_health.get("detail"),
        )

    # 启动自检 2：向量库配错的话，在这里就喊出来，而不是等第一条消息才 500
    health = await run_in_threadpool(_check_vector_backend)
    _app.state.vector_health = health
    if not health.get("ok"):
        logger.error(
            "向量库自检未通过 backend=%s collection=%s 期望维度=%s：%s\n"
            "系统仍会启动，但知识库检索会不可用。请依次检查：\n"
            "  1) QDRANT_URL / QDRANT_API_KEY 是否正确、API Key 是否失效；\n"
            "  2) 本机到该端点的网络与代理（TLS 被重置的典型症状是 UNEXPECTED_EOF；\n"
            "     注意「控制台能打开」不代表集群端点可达，控制台走的是管理面）；\n"
            "  3) Qdrant Cloud 控制台里集群是否处于运行状态（免费版可能被暂停）；\n"
            "  4) 集合已存在时，其维度必须等于 EMBED_DIM（不一致需删集合重建并重新入库）。",
            settings.vector_backend,
            health.get("collection") or "-",
            health.get("dim") or "-",
            health.get("error"),
        )

    # 启动自检 3：重排模型。**这是增强项，不通也照常启动**——
    # 检索会自动降级回纯向量排序（R@1 从 82.7% 退回 69.3%，但不会答不上来）。
    rerank_health = await rerank_healthcheck()
    _app.state.rerank_health = rerank_health
    if not rerank_health.get("ok"):
        logger.warning(
            "重排模型自检未通过（已降级为纯向量检索）provider=%s model=%s 原因=%s\n"
            "排查：RERANK_BASE_URL / RERANK_API_KEY / RERANK_MODEL 是否正确；"
            "注意 5xx 也可能是模型名写错，请以上面的原话为准。",
            rerank_health.get("provider"),
            rerank_health.get("model") or "-",
            rerank_health.get("detail"),
        )

    # 启动自检 4：视觉模型（入库时把图读成文字）。同样是增强项，不通照常启动，
    # 只是带图文档会保留「图中内容无法提取」的占位说明，正文照常入库。
    from app.vision import aclose as vision_aclose, healthcheck as vision_healthcheck

    vision_health = await vision_healthcheck()
    _app.state.vision_health = vision_health
    if not vision_health.get("ok"):
        logger.warning(
            "视觉模型自检未通过（带图文档将保留占位说明）provider=%s model=%s 原因=%s\n"
            "排查：VISION_BASE_URL / VISION_API_KEY / VISION_MODEL 是否正确；"
            "网关可能把模型名写错报成 5xx，请以上面的原话为准。",
            vision_health.get("provider"),
            vision_health.get("model") or "-",
            vision_health.get("detail"),
        )

    logger.info(
        "AICS 启动完成 env=%s llm=%s(%s) embed=%s(%s dim=%s) rerank=%s(%s) vision=%s(%s) vector=%s db=%s",
        settings.env,
        settings.llm_provider,
        llm_health.get("model") or "-",
        settings.embed_provider,
        settings.embed_model or "-",
        embed_health.get("dim") or settings.embed_dim,
        settings.rerank_provider,
        settings.rerank_model or "-",
        settings.vision_provider,
        settings.vision_model or "-",
        settings.vector_backend,
        settings.database_url,
    )
    yield

    # 释放连接池，避免 uvicorn 退出时报「未关闭的连接」
    await embed_aclose()
    await llm_aclose()
    await rerank_aclose()
    await vision_aclose()


def _check_vector_backend() -> dict:
    """向量库连通性自检（不抛异常，把结果交给调用方展示）。"""
    db = SessionLocal()
    try:
        store = build_vector_store(db)
        return store.healthcheck()
    except Exception as exc:
        # 构造 store 就失败（典型：网络/集群不可达，此时 healthcheck 还没机会执行）。
        # 仍然要带上「系统期望的目标」：集群连不上恰恰是运维最需要核对
        # 集合名与期望维度的时刻，只给一句「不可用」会让人无从下手。
        return {
            "ok": False,
            "backend": settings.vector_backend,
            "collection": (
                settings.qdrant_collection if settings.vector_backend == "qdrant" else None
            ),
            "dim": settings.embed_dim,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        db.close()


app = FastAPI(
    title="AICS · 多租户 AI 客服系统",
    version="1.0.0",
    description=(
        "多租户 AI 客服：租户隔离底座 + AI 员工配置与发布 + 知识库 RAG + "
        "意图识别与兜底 + 人工接管 + 用量计量与 per-tenant 限流。"
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# 统一异常处理：把内部异常翻译成明确的 HTTP 语义
# --------------------------------------------------------------------------- #
@app.exception_handler(TenantMismatch)
async def _tenant_mismatch(_request: Request, exc: TenantMismatch):
    # 跨租户写入或缺失租户上下文 —— 一律拒绝，不泄露任何细节
    logger.warning("租户作用域冲突已拒绝：%s", exc)
    return JSONResponse(status_code=403, content={"detail": "请求的租户上下文无效，已拒绝"})


@app.exception_handler(RateLimited)
async def _rate_limited(_request: Request, exc: RateLimited):
    return JSONResponse(
        status_code=429,
        content={"detail": exc.message},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(QuotaExceeded)
async def _quota(_request: Request, exc: QuotaExceeded):
    return JSONResponse(status_code=429, content={"detail": exc.message})


@app.exception_handler(LLMUnavailable)
async def _llm_down(_request: Request, exc: LLMUnavailable):
    return JSONResponse(
        status_code=503, content={"detail": exc.message, "reason": "llm_unavailable"}
    )


@app.exception_handler(EmbeddingUnavailable)
async def _embed_down(_request: Request, exc: EmbeddingUnavailable):
    return JSONResponse(
        status_code=503, content={"detail": exc.message, "reason": "embedding_unavailable"}
    )


@app.exception_handler(VectorStoreUnavailable)
async def _vector_down(_request: Request, exc: VectorStoreUnavailable):
    # 向量库不可用是可恢复的外部依赖故障，用 503 而不是 500，
    # 并把 exc.message 原样返回 —— 维度不匹配这类配置问题的修法就写在 message 里。
    logger.error(
        "向量库不可用 type=%s message=%s detail=%s",
        type(exc).__name__,
        exc.message,
        getattr(exc, "detail", ""),
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": exc.message,
            "reason": "vector_store_unavailable",
            "kind": type(exc).__name__,
        },
    )


app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/admin")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/widget", include_in_schema=False)
def widget_page() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "widget.html"))


@app.get("/healthz", tags=["meta"])
def healthz(request: Request) -> dict:
    vh = getattr(request.app.state, "vector_health", None)
    eh = getattr(request.app.state, "embed_health", None)
    lh = getattr(request.app.state, "llm_health", None)
    # 三个外部依赖任一不可用，客服链路就有缺口 —— 都要体现在 degraded 上。
    # 注意 ok 恒为 True（进程活着），degraded 才是「能力受损」的信号，
    # 这样探活不会因为下游抖动把实例摘掉，但运维能一眼看出降级原因。
    degraded = bool(
        (vh and not vh.get("ok")) or (eh and not eh.get("ok")) or (lh and not lh.get("ok"))
    )
    return {
        "ok": True,
        "app": settings.app_name,
        "env": settings.env,
        "llm": lh or {"ok": None, "provider": settings.llm_provider},
        "embed": eh or {"ok": None, "provider": settings.embed_provider},
        "vector": vh or {"ok": None, "backend": settings.vector_backend},
        "degraded": degraded,
    }


@app.get("/api/meta", tags=["meta"])
def meta() -> dict:
    """暴露非敏感的运行时信息与全部边界值，供前端展示与联调核对。"""
    return {
        "app": settings.app_name,
        "env": settings.env,
        "providers": {
            "llm": settings.llm_provider,
            "llm_model": settings.llm_model or "(未配置)",
            "llm_fallback_configured": bool(settings.llm_fallback_model),
            "embedding": settings.embed_provider,
            "embedding_model": settings.embed_model or "(未配置)",
            "embedding_dim": settings.embed_dim,
            "embedding_retrieval": "bge-m3 稠密向量（检索召回）",
            "vector_backend": settings.vector_backend,
            "vector_collection": settings.qdrant_collection if settings.vector_backend == "qdrant" else None,
        },
        "rag": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "min_score": settings.min_score,
        },
        "limits": {
            "persona_max_chars": settings.persona_max_chars,
            "persona_var_max_chars": settings.persona_var_max_chars,
            "max_employees_per_tenant": settings.max_employees_per_tenant,
            "max_kb_per_tenant": settings.max_kb_per_tenant,
            "max_docs_per_kb": settings.max_docs_per_kb,
            "max_kb_bytes": settings.max_kb_bytes,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_upload_files": settings.max_upload_files,
            "max_versions_kept": settings.max_versions_kept,
            "asr_max_seconds": settings.asr_max_seconds,
            "auto_stop_threshold": settings.auto_stop_threshold,
            "clarify_rounds": settings.clarify_rounds,
            "batch_test_max": settings.batch_test_max,
            "playground_keep_days": settings.playground_keep_days,
            "insight_min_samples": settings.insight_min_samples,
            "audit_keep_days": settings.audit_keep_days,
            "tenant_rpm": settings.tenant_rpm,
            "tenant_concurrent": settings.tenant_concurrent,
            "tenant_daily_tokens": settings.tenant_daily_tokens,
            "allowed_upload_ext": sorted(settings.allowed_ext),
        },
    }
