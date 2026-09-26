"""数据模型。

隔离原则（PRD 第二章）：
1. 所有租户级表都必须带 tenant_id，并建索引；
2. 所有租户级查询都必须显式带 tenant_id 条件（由 app/scoping.py 统一收口）；
3. tenant_id 只能来自服务端推导的登录态 / 渠道凭证，禁止来自请求体或查询参数。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.utils import j_load, new_id, utcnow


# --------------------------------------------------------------------------- #
# 状态常量
# --------------------------------------------------------------------------- #
class TenantStatus:
    ACTIVE = "active"
    SUSPENDED = "suspended"


class UserRole:
    PLATFORM_ADMIN = "platform_admin"   # 平台运营，可跨租户
    TENANT_ADMIN = "tenant_admin"       # 租户管理员
    CONFIG_EDITOR = "config_editor"     # AI 配置者
    AGENT = "agent"                     # 人工客服
    VIEWER = "viewer"                   # 只读

    TENANT_ROLES = (TENANT_ADMIN, CONFIG_EDITOR, AGENT, VIEWER)


class EmployeeStatus:
    DRAFT = "draft"
    PUBLISHED = "published"


class LowConfidencePolicy:
    CLARIFY = "clarify"      # 先追问澄清，超过轮次转人工
    HANDOFF = "handoff"      # 直接转人工
    FALLBACK = "fallback"    # 兜底话术


class GroupReplyPolicy:
    MENTION_ONLY = "mention_only"
    ALL = "all"


class DocStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class SessionStatus:
    AI = "ai"
    QUEUED = "queued"     # 待人工接入
    HUMAN = "human"       # 人工接管中
    STOPPED = "stopped"   # 自动停止回复（轮次/时间窗触发）；新会话自动恢复
    CLOSED = "closed"


class MessageRole:
    VISITOR = "visitor"
    AI = "ai"
    AGENT = "agent"
    SYSTEM = "system"


# --------------------------------------------------------------------------- #
# 基础 Mixin
# --------------------------------------------------------------------------- #
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class TenantScopedMixin:
    """租户级表统一带 tenant_id（由各表显式声明，见下方 tenant_id 字段）。"""


# --------------------------------------------------------------------------- #
# 平台 / 租户 / 账号
# --------------------------------------------------------------------------- #
class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default=TenantStatus.ACTIVE, nullable=False)
    plan: Mapped[str] = mapped_column(String(16), default="free", nullable=False)
    # 配额：租户级，超限只影响本租户
    daily_token_quota: Mapped[int] = mapped_column(Integer, default=2_000_000, nullable=False)
    rpm_limit: Mapped[int] = mapped_column(Integer, default=120, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Shanghai", nullable=False)
    default_language: Mapped[str] = mapped_column(String(16), default="zh-CN", nullable=False)
    note: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[str] = mapped_column(Text, default="{}")

    users: Mapped[list["User"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tenant_role", "tenant_id", "role"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # 平台运营为 NULL；租户账号必须有值
    tenant_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("tenants.id"), nullable=True, index=True
    )
    email: Mapped[str] = mapped_column(String(190), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(80), default="")
    role: Mapped[str] = mapped_column(String(32), default=UserRole.VIEWER, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    tenant: Mapped["Tenant | None"] = relationship(back_populates="users")


class WidgetKey(Base, TimestampMixin):
    """渠道凭证：访客侧靠它推导租户与 AI 员工（PRD 2.5「按渠道凭证绑定关系推导租户」）。"""

    __tablename__ = "widget_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("ai_employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="网页插件")
    channel: Mapped[str] = mapped_column(String(32), default="web", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)


# --------------------------------------------------------------------------- #
# AI 员工配置
# --------------------------------------------------------------------------- #
class AiEmployee(Base, TimestampMixin):
    __tablename__ = "ai_employees"
    __table_args__ = (Index("ix_employee_tenant_default", "tenant_id", "is_default"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 草稿态字段：发布时快照进 employee_versions
    status: Mapped[str] = mapped_column(String(16), default=EmployeeStatus.DRAFT, nullable=False)
    published_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    persona: Mapped[str] = mapped_column(Text, default="")
    output_language: Mapped[str] = mapped_column(String(32), default="auto", nullable=False)
    fast_mode: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    memory_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    auto_stop_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    auto_stop_threshold: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    group_reply_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    group_reply_policy: Mapped[str] = mapped_column(
        String(24), default=GroupReplyPolicy.ALL, nullable=False
    )
    asr_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    time_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    low_confidence_policy: Mapped[str] = mapped_column(
        String(16), default=LowConfidencePolicy.CLARIFY, nullable=False
    )
    confidence_threshold: Mapped[int] = mapped_column(Integer, default=35, nullable=False)  # 0-100
    clarify_rounds: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    llm_temperature: Mapped[int] = mapped_column(Integer, default=30, nullable=False)  # 存 0-100
    # 推理模型（如 deepseek-v4-flash）的思考 token 也占这份预算，默认给宽一点；
    # 撞上限会截成半句回答，宁可多给额度也不要给半句话。上限见 schemas.py。
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=1200, nullable=False)
    # 渠道级 AI 开关：{"web": true, "email": false}
    # 本期界面已下线该配置，字段与 handle_turn 的判定保留仅为兼容存量数据（默认全放行）。
    channel_switch: Mapped[str] = mapped_column(Text, default='{"web":true}')

    # ---- AI 回复设置：拟人 / 延迟 / 拆分 ----
    humanize_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reply_delay_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 旧字段：固定延迟，取区间 max 同步
    # 延迟回复区间（秒）：每次回复在 [min, max] 内随机；min=max=0 表示关闭
    reply_delay_min: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reply_delay_max: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    split_reply_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    split_reply_max: Mapped[int] = mapped_column(Integer, default=3, nullable=False)  # 2-5 条
    split_reply_interval_ms: Mapped[int] = mapped_column(Integer, default=800, nullable=False)  # 旧字段：固定间隔，取区间 min 同步
    # 拆分消息发送间隔区间（ms）：每条间隔在 [min, max] 内随机
    split_interval_min_ms: Mapped[int] = mapped_column(Integer, default=800, nullable=False)
    split_interval_max_ms: Mapped[int] = mapped_column(Integer, default=800, nullable=False)

    # ---- 自动停止回复（轮次 / 时间窗）。与 auto_stop_*（未有效回答转人工）并存 ----
    stop_reply_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    stop_reply_rounds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 旧字段：条件组为空时回退用
    stop_reply_message: Mapped[str] = mapped_column(String(500), default="", nullable=False)  # 结束语，可空
    # 停止条件组：[[{"type":"ai_rounds|visitor_msgs","op":"gte","value":N}, ...], ...]
    # 组内条件「且」，组间「或」；任一组全部满足即触发停止。为空时回退 stop_reply_rounds。
    stop_condition_groups: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    stop_time_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # [{"days":[0..6（0=周一）], "start":"HH:MM", "end":"HH:MM"}]；end<=start 视为跨零点
    stop_time_rules: Mapped[str] = mapped_column(Text, default="[]", nullable=False)


class EmployeeVersion(Base):
    __tablename__ = "employee_versions"
    __table_args__ = (
        UniqueConstraint("employee_id", "version_no", name="uq_employee_version"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("ai_employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[str] = mapped_column(Text, default="{}")
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    @property
    def data(self) -> dict:
        return j_load(self.snapshot, {}) or {}


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_kb_tenant_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    total_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class EmployeeKbBinding(Base):
    __tablename__ = "employee_kb_bindings"
    __table_args__ = (UniqueConstraint("employee_id", "kb_id", name="uq_employee_kb"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("ai_employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )


class EmployeeDocBinding(Base):
    """AI 员工 ↔ 文档 绑定（文件级检索范围，优先于知识库绑定）。"""

    __tablename__ = "employee_doc_bindings"
    __table_args__ = (UniqueConstraint("employee_id", "doc_id", name="uq_employee_doc"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("ai_employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doc_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_doc_tenant_kb", "tenant_id", "kb_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    ext: Mapped[str] = mapped_column(String(16), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=DocStatus.PENDING, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), default="upload", nullable=False)


class AssetStatus:
    """文档内嵌图片的 OCR/描述状态。

    与 DocStatus 分开：图片处理失败只是「这张图没读出来」，
    文档本身仍然入库成功 —— 不能用一个失败态把整篇文档打成失败。
    """

    PENDING = "pending"   # 已抽出，排队等视觉模型
    OK = "ok"             # 已转成文字并回写占位块
    FAILED = "failed"     # 模型调用失败，保留原占位符
    SKIPPED = "skipped"   # 未启用图片识别，或判定为装饰性小图


class DocumentAsset(Base, TimestampMixin):
    """文档里抽出的一张图，以及它对块的归属。

    为什么要留这一行（而不只是把识别结果塞进正文就完事）：
    1. **可核对**：模型识别可能出错（实测过把 MTK7621+7612 认成 MTK7621+761），
       留有原文和原图，出错时能追溯是哪张图。
    2. **可重跑**：换视觉模型或调提示词后，不用重新上传文档就能重做。
    3. **可召回**：后续要支持「回答时把原图一并发出去」，图必须先有落盘和 URL。
    """

    __tablename__ = "document_assets"
    __table_args__ = (Index("ix_asset_tenant_doc", "tenant_id", "doc_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # 在文档中的出现序号（从 1 起），与占位块的出现顺序一一对应
    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    mime: Mapped[str] = mapped_column(String(32), default="image/png", nullable=False)
    ext: Mapped[str] = mapped_column(String(8), default="png", nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 相对 assets_dir 的路径，不存绝对路径（迁移机器时不至于全盘失效）
    rel_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    # 视觉模型读出的正文；为空表示这张图没读出来
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default=AssetStatus.PENDING, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="")


class Chunk(Base, TimestampMixin):
    """向量以 JSON 存储，供 VECTOR_BACKEND=db 时暴力检索；换 qdrant/pgvector 时此表仍保留原文。"""

    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunk_tenant_kb", "tenant_id", "kb_id", "enabled"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    embedding: Mapped[str] = mapped_column(Text, default="")


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
class Session(Base, TimestampMixin):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_session_tenant_status", "tenant_id", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(32), default="web", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=SessionStatus.AI, nullable=False)
    assigned_user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    clarify_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    no_answer_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    handoff_reason: Mapped[str] = mapped_column(String(64), default="")
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    meta: Mapped[str] = mapped_column(Text, default="{}")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_msg_tenant_session", "tenant_id", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    @property
    def meta_data(self) -> dict:
        return j_load(self.meta, {}) or {}


class VisitorMemory(Base, TimestampMixin):
    """AI 记忆：按租户 + 访客双维度隔离。"""

    __tablename__ = "visitor_memories"
    __table_args__ = (Index("ix_memory_tenant_visitor", "tenant_id", "visitor_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(24), default="fact", nullable=False)


# --------------------------------------------------------------------------- #
# 计量 / 审计 / 测试
# --------------------------------------------------------------------------- #
class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (Index("ix_usage_tenant_time", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(16), default="llm", nullable=False)  # llm|embedding|rerank|asr
    model: Mapped[str] = mapped_column(String(80), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    origin: Mapped[str] = mapped_column(String(24), default="chat", nullable=False)  # chat|playground|ingest
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_tenant_time", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_email: Mapped[str] = mapped_column(String(190), default="")
    action: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    target: Mapped[str] = mapped_column(String(190), default="")
    result: Mapped[str] = mapped_column(String(16), default="allow", nullable=False)  # allow|deny
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class PlaygroundRun(Base):
    __tablename__ = "playground_runs"
    __table_args__ = (Index("ix_playground_tenant_time", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(32), ForeignKey("tenants.id"), nullable=False, index=True)
    employee_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    hits: Mapped[str] = mapped_column(Text, default="[]")
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ModelEndpoint(Base, TimestampMixin):
    """平台级模型端点（PRD 2.2：模型接入凭证平台共享、租户不可见）。"""

    __tablename__ = "model_endpoints"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="llm", nullable=False)  # llm|embedding|rerank
    base_url: Mapped[str] = mapped_column(String(255), default="")
    model: Mapped[str] = mapped_column(String(120), default="")
    api_key_masked: Mapped[str] = mapped_column(String(64), default="")
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
