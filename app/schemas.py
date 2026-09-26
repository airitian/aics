"""请求/响应模型。边界值校验集中在这里（对应 PRD 第八章）。"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.models import GroupReplyPolicy, LowConfidencePolicy

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

# 自动停止回复：条件类型与比较符白名单（op 预留扩展，当前仅支持 gte=大于等于）
_STOP_COND_TYPES = {"ai_rounds", "visitor_msgs"}
_STOP_COND_OPS = {"gte"}


def _validate_stop_groups(v: list[list[dict]]) -> list[list[dict]]:
    """停止条件组：组内「且」，组间「或」。每组至少一个条件，条件字段白名单校验。"""
    for group in v or []:
        if not isinstance(group, list) or not group:
            raise ValueError("每个条件组至少包含一个条件")
        for cond in group:
            if not isinstance(cond, dict):
                raise ValueError("条件必须是对象")
            if cond.get("type") not in _STOP_COND_TYPES:
                raise ValueError(f"不支持的停止条件类型：{cond.get('type')}")
            if str(cond.get("op") or "gte") not in _STOP_COND_OPS:
                raise ValueError("停止条件目前仅支持「大于等于」")
            val = cond.get("value")
            if isinstance(val, bool) or not isinstance(val, int) or not 1 <= val <= 999:
                raise ValueError("停止条件阈值必须是 1-999 的整数")
    return v

SUPPORTED_LANGUAGES = {
    "auto",
    "zh-CN",
    "zh-TW",
    "en",
    "ja",
    "ko",
    "ru",
    "es",
    "pt",
    "fr",
    "de",
    "ar",
    "th",
    "vi",
    "id",
}


class ApiMessage(BaseModel):
    ok: bool = True
    message: str = ""


# --------------------------------------------------------------------------- #
# 认证 / 租户 / 账号
# --------------------------------------------------------------------------- #
class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=190)
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: str | None
    tenant_name: str | None = None
    name: str = ""
    email: str = ""


class TenantCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    admin_email: str = Field(min_length=3, max_length=190)
    admin_password: str = Field(min_length=8, max_length=128)
    plan: Literal["free", "standard", "pro"] = "free"
    daily_token_quota: int = Field(default=2_000_000, ge=10_000, le=10_000_000_000)
    rpm_limit: int = Field(default=120, ge=1, le=100_000)
    timezone: str = Field(default="Asia/Shanghai", max_length=64)
    default_language: str = Field(default="zh-CN", max_length=16)


class TenantOut(BaseModel):
    id: str
    name: str
    slug: str
    status: str
    plan: str
    daily_token_quota: int
    rpm_limit: int
    timezone: str
    default_language: str
    created_at: Any = None


class TenantUpdateIn(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    status: Literal["active", "suspended"] | None = None
    plan: Literal["free", "standard", "pro"] | None = None
    daily_token_quota: int | None = Field(default=None, ge=10_000)
    rpm_limit: int | None = Field(default=None, ge=1, le=100_000)
    timezone: str | None = Field(default=None, max_length=64)
    default_language: str | None = Field(default=None, max_length=16)


class UserCreateIn(BaseModel):
    email: str = Field(min_length=3, max_length=190)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=80)
    role: Literal["tenant_admin", "config_editor", "agent", "viewer"] = "agent"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    status: str


# --------------------------------------------------------------------------- #
# AI 员工
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    is_default: bool = False
    persona: str = Field(default="", max_length=settings.persona_max_chars)
    output_language: str = "auto"
    fast_mode: bool = True
    memory_enabled: bool = False
    auto_stop_enabled: bool = False
    auto_stop_threshold: int = Field(default=3, ge=1, le=10)
    group_reply_enabled: bool = True
    group_reply_policy: str = GroupReplyPolicy.ALL
    asr_enabled: bool = True
    time_enabled: bool = True
    low_confidence_policy: str = LowConfidencePolicy.CLARIFY
    confidence_threshold: int = Field(default=35, ge=0, le=100)
    clarify_rounds: int = Field(default=1, ge=0, le=3)
    llm_temperature: int = Field(default=30, ge=0, le=100)
    llm_max_tokens: int = Field(default=1200, ge=1, le=4096)
    channel_switch: dict[str, bool] = Field(default_factory=lambda: {"web": True})
    kb_ids: list[str] = Field(default_factory=list)
    # 文件级绑定：优先于 kb_ids；绑定后检索范围 = 这些文档的片段
    doc_ids: list[str] = Field(default_factory=list)
    # ---- AI 回复设置 ----
    humanize_enabled: bool = False
    reply_delay_seconds: int = Field(default=0, ge=0, le=60)  # 旧字段：固定延迟，取区间 max 同步
    reply_delay_min: int = Field(default=0, ge=0, le=60)
    reply_delay_max: int = Field(default=0, ge=0, le=60)
    split_reply_enabled: bool = False
    split_reply_max: int = Field(default=3, ge=2, le=5)
    split_reply_interval_ms: int = Field(default=800, ge=100, le=5000)  # 旧字段：固定间隔，取区间 min 同步
    split_interval_min_ms: int = Field(default=800, ge=100, le=5000)
    split_interval_max_ms: int = Field(default=800, ge=100, le=5000)

    @model_validator(mode="after")
    def _check_delay_and_split_ranges(self) -> "EmployeeIn":
        if self.reply_delay_min > self.reply_delay_max:
            raise ValueError("延迟回复区间最小值不能大于最大值")
        if self.split_interval_min_ms > self.split_interval_max_ms:
            raise ValueError("拆分发送间隔区间最小值不能大于最大值")
        return self
    # ---- 自动停止回复（条件组/时间窗）----
    stop_reply_enabled: bool = False
    stop_reply_rounds: int = Field(default=0, ge=0, le=100)  # 旧字段，仅作回退
    stop_condition_groups: list[list[dict]] = Field(default_factory=list)
    stop_reply_message: str = Field(default="", max_length=500)
    stop_time_enabled: bool = False
    stop_time_rules: list[dict] = Field(default_factory=list)

    @field_validator("stop_condition_groups")
    @classmethod
    def _check_stop_groups_in(cls, v: list[list[dict]]) -> list[list[dict]]:
        return _validate_stop_groups(v)

    @field_validator("stop_time_rules")
    @classmethod
    def _check_stop_rules_in(cls, v: list[dict]) -> list[dict]:
        if v:
            for rule in v:
                days = rule.get("days")
                start, end = rule.get("start"), rule.get("end")
                if not isinstance(days, list) or not days or not all(
                    isinstance(d, int) and 0 <= d <= 6 for d in days
                ):
                    raise ValueError("停止时间的 days 必须是 0-6（0=周一）的非空数组")
                if not _HHMM_RE.match(str(start or "")) or not _HHMM_RE.match(str(end or "")):
                    raise ValueError("停止时间必须为 HH:MM 格式")
        return v

    @field_validator("output_language")
    @classmethod
    def _check_lang(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的输出语言：{v}")
        return v

    @field_validator("group_reply_policy")
    @classmethod
    def _check_group(cls, v: str) -> str:
        if v not in (GroupReplyPolicy.ALL, GroupReplyPolicy.MENTION_ONLY):
            raise ValueError("群聊回复策略只能是 all 或 mention_only")
        return v

    @field_validator("low_confidence_policy")
    @classmethod
    def _check_low_conf(cls, v: str) -> str:
        allowed = (
            LowConfidencePolicy.CLARIFY,
            LowConfidencePolicy.HANDOFF,
            LowConfidencePolicy.FALLBACK,
        )
        if v not in allowed:
            raise ValueError("低置信度策略只能是 clarify / handoff / fallback")
        return v


class EmployeePatchIn(BaseModel):
    """局部更新，字段全部可选。"""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    is_default: bool | None = None
    persona: str | None = Field(default=None, max_length=settings.persona_max_chars)
    output_language: str | None = None
    fast_mode: bool | None = None
    memory_enabled: bool | None = None
    auto_stop_enabled: bool | None = None
    auto_stop_threshold: int | None = Field(default=None, ge=1, le=10)
    group_reply_enabled: bool | None = None
    group_reply_policy: str | None = None
    asr_enabled: bool | None = None
    time_enabled: bool | None = None
    low_confidence_policy: str | None = None
    confidence_threshold: int | None = Field(default=None, ge=0, le=100)
    clarify_rounds: int | None = Field(default=None, ge=0, le=3)
    llm_temperature: int | None = Field(default=None, ge=0, le=100)
    llm_max_tokens: int | None = Field(default=None, ge=1, le=4096)
    channel_switch: dict[str, bool] | None = None
    kb_ids: list[str] | None = None
    doc_ids: list[str] | None = None
    # ---- AI 回复设置 ----
    humanize_enabled: bool | None = None
    reply_delay_seconds: int | None = Field(default=None, ge=0, le=60)
    reply_delay_min: int | None = Field(default=None, ge=0, le=60)
    reply_delay_max: int | None = Field(default=None, ge=0, le=60)
    split_reply_enabled: bool | None = None
    split_reply_max: int | None = Field(default=None, ge=2, le=5)
    split_reply_interval_ms: int | None = Field(default=None, ge=100, le=5000)
    split_interval_min_ms: int | None = Field(default=None, ge=100, le=5000)
    split_interval_max_ms: int | None = Field(default=None, ge=100, le=5000)

    @model_validator(mode="after")
    def _check_delay_and_split_ranges_patch(self) -> "EmployeePatchIn":
        # 部分更新：仅当区间两端同时给出时才校验，单端交给「旧值 + 新值」合并后仍需合法
        lo = self.reply_delay_min if self.reply_delay_min is not None else self.reply_delay_max
        hi = self.reply_delay_max if self.reply_delay_max is not None else self.reply_delay_min
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("延迟回复区间最小值不能大于最大值")
        slo = self.split_interval_min_ms if self.split_interval_min_ms is not None else self.split_interval_max_ms
        shi = self.split_interval_max_ms if self.split_interval_max_ms is not None else self.split_interval_min_ms
        if slo is not None and shi is not None and slo > shi:
            raise ValueError("拆分发送间隔区间最小值不能大于最大值")
        return self

    # ---- 自动停止回复 ----
    stop_reply_enabled: bool | None = None
    stop_reply_rounds: int | None = Field(default=None, ge=0, le=100)
    stop_condition_groups: list[list[dict]] | None = None
    stop_reply_message: str | None = Field(default=None, max_length=500)
    stop_time_enabled: bool | None = None
    stop_time_rules: list[dict] | None = None

    @field_validator("stop_condition_groups")
    @classmethod
    def _check_stop_groups(cls, v: list[list[dict]] | None) -> list[list[dict]] | None:
        if v is None:
            return v
        return _validate_stop_groups(v)

    @field_validator("stop_time_rules")
    @classmethod
    def _check_stop_rules(cls, v: list[dict] | None) -> list[dict] | None:
        if v is None:
            return v
        for rule in v:
            days = rule.get("days")
            start, end = rule.get("start"), rule.get("end")
            if not isinstance(days, list) or not days or not all(
                isinstance(d, int) and 0 <= d <= 6 for d in days
            ):
                raise ValueError("停止时间的 days 必须是 0-6（0=周一）的非空数组")
            if not _HHMM_RE.match(str(start or "")) or not _HHMM_RE.match(str(end or "")):
                raise ValueError("停止时间必须为 HH:MM 格式")
        return v

    @field_validator("output_language")
    @classmethod
    def _check_lang(cls, v: str | None) -> str | None:
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的输出语言：{v}")
        return v


class PublishIn(BaseModel):
    """发布：人设为空等阻断项由服务端校验；force 仅用于跳过「警告」级校验。"""

    force: bool = False


class PersonaDraftIn(BaseModel):
    """人设智能生成 / 优化。generate 用 description，optimize 用 current_persona。"""

    mode: Literal["generate", "optimize"] = "generate"
    description: str = Field(default="", max_length=2000)
    optimize_note: str = Field(default="", max_length=1000)
    current_persona: str = Field(default="", max_length=settings.persona_max_chars)
    # 快捷设定：语气风格 / 回复长度，留空 = 不限定
    style: str = Field(default="", max_length=20)
    length: str = Field(default="", max_length=20)
    output_language: str = Field(default="zh-CN", max_length=16)


class PublishResultOut(BaseModel):
    ok: bool
    version_no: int = 0
    blocked: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str = ""


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
class KbIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class KbOut(BaseModel):
    id: str
    name: str
    description: str
    doc_count: int = 0
    chunk_count: int = 0
    total_bytes: int = 0


class DocOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    ext: str
    size_bytes: int
    status: str
    error: str = ""
    chunk_count: int
    enabled: bool
    created_at: Any = None


class SearchTestIn(BaseModel):
    kb_id: str | None = None
    employee_id: str | None = None
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    visitor_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=32)
    channel: str = Field(default="web", max_length=32)
    # 本条消息携带的聊天图片（先经 /api/chat/images 上传拿 id）。
    # 上限在 chatimages.resolve_and_transcribe 里按配置截断并校验归属。
    image_ids: list[str] = Field(default_factory=list)


class HitOut(BaseModel):
    chunk_id: str
    doc_id: str
    kb_id: str
    filename: str = ""
    score: float
    text: str


class ChatOut(BaseModel):
    session_id: str
    reply: str
    status: str
    handoff: bool
    handoff_reason: str = ""
    hits: list[HitOut] = Field(default_factory=list)
    degraded: bool = False
    degrade_reason: str = ""
    usage_total_tokens: int = 0
    # 消息拆分回复：开启拆分且回答较长时，分段返回（reply 仍是完整文本）
    segments: list[str] = Field(default_factory=list)


class AgentReplyIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class SessionOut(BaseModel):
    id: str
    employee_id: str
    visitor_id: str
    channel: str
    status: str
    handoff_reason: str = ""
    assigned_user_id: str | None = None
    no_answer_streak: int = 0
    last_active_at: Any = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: Any = None
    meta: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 问答测试
# --------------------------------------------------------------------------- #
class PlaygroundIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class BatchTestIn(BaseModel):
    questions: list[str] = Field(default_factory=list)


class PlaygroundOut(BaseModel):
    question: str
    answer: str
    hits: list[HitOut] = Field(default_factory=list)
    top_score: float = 0.0
    latency_ms: int = 0
    tokens: int = 0
    ok: bool = True
    error: str = ""


# --------------------------------------------------------------------------- #
# AI 员工测试（聊天式多轮对话）
# --------------------------------------------------------------------------- #
class TestChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # 对话测试也支持带图（与线上同一套图片链路，见 /api/testchat/{eid}/images）
    image_ids: list[str] = Field(default_factory=list)


class TestChatOut(BaseModel):
    """一轮测试对话的结果。

    除回复本身外，把「为什么这么答」的依据一并返回：命中片段、置信度、
    耗时、token、是否降级/转人工。测试页要能一眼看出答错是检索问题还是
    模型问题，只给一句回复是做不到的。
    """

    session_id: str
    reply: str
    status: str
    handoff: bool = False
    handoff_reason: str = ""
    hits: list[HitOut] = Field(default_factory=list)
    top_score: float = 0.0
    confidence: int = 0
    latency_ms: int = 0
    tokens: int = 0
    degraded: bool = False
    degrade_reason: str = ""
