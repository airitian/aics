"""全局配置：一切可变量走环境变量，代码中不出现密钥。"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ---------- 基础 ----------
    app_name: str = "AICS"
    env: Literal["dev", "prod", "test"] = "dev"
    debug: bool = False
    database_url: str = "sqlite:///./aics.db"
    static_dir: str = "app/static"

    jwt_secret: str = "dev-secret-change-me"
    jwt_alg: str = "HS256"
    access_token_ttl_minutes: int = 720

    bootstrap_admin_email: str = "admin@aics.local"
    bootstrap_admin_password: str = "admin12345"
    bootstrap_demo: bool = True
    # 免登录模式：开启后管理后台跳过登录页，/api/auth/auto 自动签发默认租户管理员令牌。
    # 仅限演示/内网环境，生产务必关闭。
    auth_disabled: bool = False

    # ---------- 对话模型 ----------
    llm_provider: Literal["api", "stub"] = "stub"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.3
    llm_max_tokens: int = 800
    # 推理预算（reasoning_effort）：客服场景默认压到 minimal。
    # 思考开关（DeepSeek 官方 API 的 thinking 参数）：enabled / disabled / ""（不传）。
    # 实测（DeepSeek-V4.1-Flash，同一句开放性提问）：
    #   thinking.enabled  → 7s，且思维链会把 max_tokens 吃光，正文返回 0 字（空答）
    #   thinking.disabled → 1.0s，思维 0 字，正文正常
    # 客服是实时对话、且答案来自检索到的资料，不需要模型自己深想，所以默认关掉。
    llm_thinking: str = "disabled"
    # 推理档位（官方仅支持 low / high / max，传 minimal 会被静默忽略并退回 high）。
    # 只在 thinking=enabled 时才有意义；置 "" 则不向上传参。
    # 实测：推理模型在"资料没命中/问题开放"时会展开上万字思维链——同一句开放性提问
    # 默认 54s、思维链 14616 字。客服要的是快答，不是深度思考。
    llm_reasoning_effort: str = "low"
    # 连续失败达到阈值后熔断：冷却期内直接失败，避免每条客户消息都白等一轮退避
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown: float = 15.0
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""
    # ---------- 开发测试专用 LLM 档位（profile="dev"）----------
    # 用途区分：网页端「点击测试」（testchat，不带 X-LLM-Profile 头）固定走上面的
    # 官方主档；自动化开发测试带 X-LLM-Profile: dev 头，走这组旧中转端点省钱。
    # 三项任一为空 = dev 档未配置，此时 dev 请求自动回落主档（不会报错）。
    llm_dev_base_url: str = ""
    llm_dev_api_key: str = ""
    llm_dev_model: str = ""
    # dev 档的思维链控制：置 "" 则不向上游传 thinking/reasoning_effort 参数。
    # 旧中转（PackyAPI）未验证过这两个参数的透传行为，默认不传，保持其历史可用配置。
    llm_dev_thinking: str = ""
    llm_dev_reasoning_effort: str = ""

    # ---------- 向量模型 ----------
    # api   = 调 OpenAI 兼容 /embeddings（生产）。本项目接入 Gitee AI 模力方舟的 bge-m3。
    # local = 本地确定性哈希向量，仅供离线开发/单测跑通链路，无真实语义能力。
    embed_provider: Literal["api", "local"] = "local"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    # 必须等于 EMBED_MODEL 的输出维度（bge-m3 = 1024）。
    # 向量库集合维度一经创建即固定，改了这里就必须删集合重建 + 重新入库。
    embed_dim: int = 1024
    embed_batch: int = 32
    embed_timeout: float = 30.0
    # 向量服务也在公网上，抖动会让入库/检索整条链路失败，同样是退避重试 + 熔断
    embed_retries: int = 3
    embed_retry_delay: float = 0.5
    embed_breaker_threshold: int = 3
    embed_breaker_cooldown: float = 15.0

    # ---------- 向量库 ----------
    vector_backend: Literal["db", "qdrant", "pgvector"] = "db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "aics_chunks"
    qdrant_timeout: float = 20.0
    # 云端向量库在公网上，TLS 抖动会直接让一条客户消息降级，所以默认重试 2 次
    qdrant_retries: int = 3
    qdrant_retry_delay: float = 0.5
    # 连续失败达到阈值后进入冷却：冷却期内直接降级，不再逐请求白等重试
    qdrant_breaker_threshold: int = 3
    qdrant_breaker_cooldown: float = 15.0
    pgvector_table: str = "aics_vectors"

    # ---------- 重排模型（二阶段检索的第二段） ----------
    # off = 只用向量粗排；api = 粗排取候选后交给交叉编码器精排。
    # 实测（verify_retrieval.py，75 题）：R@1 69.3% → 82.7%，是检索质量最大的单一杠杆，
    # 远大于换 embedding 模型（+2.6pp，落在噪声内）。
    # 重排是**增强项**：它挂了会降级回向量排序，不会让会话答不上来。
    rerank_provider: Literal["off", "api"] = "off"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    # 粗排候选数：rerank 逐对计算，候选越多越慢。20 条是质量/延迟的平衡点。
    # 注意线上 TOP_K=5 是**精排后**的条数，粗排必须放宽，否则答案常在第 6 名开外
    # 而根本没进候选池 —— 这正是原来 R@5 卡在 86.7% 的原因。
    rerank_candidates: int = 20
    # 粗排阶段的相似度下限。**必须低于 MIN_SCORE**：有 rerank 把关时，
    # 粗排的唯一职责是「别把正确答案漏在候选池外」，宁可多捞一点噪声让精排去筛。
    # 若沿用 MIN_SCORE=0.45，答案常在第一轮就被砍掉，rerank 再强也排不出来。
    rerank_recall_min_score: float = 0.15
    rerank_timeout: float = 20.0
    rerank_retries: int = 2
    rerank_retry_delay: float = 0.5
    rerank_breaker_threshold: int = 3
    rerank_breaker_cooldown: float = 15.0
    # 精排后的绝对下限，挡掉完全离题的候选。
    # 注意量纲与余弦不同：交叉编码器给相关项 0.6~0.99、离题项常低于 1e-3，
    # 所以这个阈值比 MIN_SCORE 低一个数量级，不要用 0.45 去套。
    rerank_min_score: float = 0.05
    # >0 时作为上游 top_n 参数（让服务端只返回前 N 条），0 = 不传
    rerank_top_n: int = 0

    # ---------- 图片识别：入库阶段把「图」读成「文」 ----------
    # 目的不是以图搜图，而是让「答案藏在图里」的问题也能被检索到。
    # 视觉能力只在**入库时**用一次，产出的就是普通文本，检索链路完全不用动。
    # off = 保留原占位符（AI 会如实说读不到图中内容，不猜）
    # api = 抽出图片后交给视觉模型转写，写回占位块再入库
    vision_provider: Literal["off", "api"] = "off"
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    vision_timeout: float = 60.0
    vision_max_tokens: int = 1200
    # 单图上限：超过就不送模型，省得一张几 MB 的插图烧光配额
    vision_max_bytes: int = 5 * 1024 * 1024
    # 单文档最多处理多少张图 —— 防止一份产品手册把入库卡成几分钟
    vision_max_images: int = 20
    # 并发上限：视觉模型慢且贵，全并发会触发上游限流，反而一张都读不出来
    vision_concurrency: int = 4
    vision_retries: int = 2
    vision_retry_delay: float = 0.5
    vision_breaker_threshold: int = 3
    vision_breaker_cooldown: float = 30.0
    # 扫描版 PDF 页面转写（无文字层兜底）：把整页渲染成图片送视觉模型抄字。
    # 整页文字比单张插图密得多，给更高的 token 上限，否则抄到一半被截断
    # （max_tokens 截断的历史教训：思维链/长文都容易烧穿小配额）。
    vision_page_max_tokens: int = 3000
    # 渲染 DPI：150 在「小字可读」和「图片体积/上传耗时」之间的平衡点
    pdf_ocr_dpi: int = 150
    # 单文档最多转写多少个扫描页 —— 上传是同步的，页数无上限会把接口卡成十分钟
    pdf_ocr_max_pages: int = 30
    # 原文件与抽出图片的存放目录。原文件必须落盘：解析完成即丢弃的话，
    # 换切分策略或补做 OCR 时就再也没机会了。
    assets_dir: str = "assets"

    # ---------- 聊天图片：客户在对话中发图 ----------
    # 客户上传图片 → 落盘 → 发送消息时视觉转写 → 转写文本进 prompt + 检索。
    # 单张上限：手机拍照原图能到 8-12MB，10MB 覆盖绝大多数截图/照片，
    # 再大的让客户裁一裁（也防几张图把回合延迟拖到分钟级）。
    chat_image_max_bytes: int = 10 * 1024 * 1024
    # 单条消息最多带几张图 —— 每张都要过一次视觉模型，张数直接乘在回合延迟上
    chat_image_max_per_message: int = 3

    # ---------- RAG ----------
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5
    min_score: float = 0.20
    # 检索隔离：绑定多个知识库时按库独立检索，每库至少取这么多条候选
    # （再全局排序去重、总量仍受 top_k 约束），避免大库把小库的高分段挤出去
    per_kb_top_k: int = 2
    # RAG 引擎：legacy = 自研向量管线；llama = LlamaIndex 标准化引擎
    # （QdrantVectorStore + BM25 jieba + QueryFusionRetriever RRF 双路召回），
    # 独立集合 aics_chunks_v2，双轨并行、基线达标后切换。
    rag_engine: Literal["legacy", "llama"] = "legacy"
    qdrant_collection_llama: str = "aics_chunks_v2"

    # ---------- 边界值（PRD 第八章） ----------
    persona_max_chars: int = 20000
    persona_var_max_chars: int = 200
    max_employees_per_tenant: int = 10
    max_kb_per_tenant: int = 20
    max_docs_per_kb: int = 500
    max_kb_bytes: int = 1024 ** 3
    max_upload_bytes: int = 20 * 1024 ** 2
    max_upload_files: int = 10
    max_versions_kept: int = 10
    auto_save_interval_sec: int = 30
    asr_max_seconds: int = 60
    auto_stop_threshold: int = 3
    clarify_rounds: int = 1
    batch_test_max: int = 50
    playground_keep_days: int = 7
    insight_min_samples: int = 30
    audit_keep_days: int = 180

    # ---------- per-tenant 限流与配额 ----------
    tenant_rpm: int = 120
    tenant_concurrent: int = 8
    tenant_daily_tokens: int = 2_000_000

    allowed_upload_ext: str = ".pdf,.docx,.txt,.md,.csv,.xlsx,.xls"

    @property
    def allowed_ext(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_upload_ext.split(",") if e.strip()}

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def plan_tier_names(self) -> list[str]:
        return ["free", "standard", "pro"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# --------------------------------------------------------------------------- #
# 「AI 员工测试」专用渠道标识
# --------------------------------------------------------------------------- #
# 测试会话落的是真实的 sessions / messages 行（否则就没有多轮上下文），
# 所以必须有一个字段能把它们和线上访客会话区分开，否则：
#   1) 概览的会话总数/拦截率/转人工会被测试数据污染，运营看到的数字是假的；
#   2) 会话入口会把测试会话混进访客列表。
# channel 是现成的区分维度，这里统一取值，供写入方（testchat）与
# 过滤方（insights）共用 —— 两处各写一遍字符串字面量迟早会写歪。
TEST_CHANNEL = "playground"
