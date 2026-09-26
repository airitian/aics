"""对话编排：一次访客消息 → 一条回复。

决策顺序（PRD 第四/六章）：
1. 渠道是否允许 AI 接待 → 不允许则直接转人工
2. 自动停止检查：停止时段内 / 会话已停止 → 静默；回复轮次达阈值 → 发结束语后停止
3. 客户主动要求转人工 / 高风险话题 → 强制转人工
4. 无意义输入 → 主动询问需求
5. 检索 + 置信度 → 无命中或低置信走配置的兜底策略
6. 生成回复；模型不可用时**明确告知并转人工**，绝不编造
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app import intent as intent_mod
from app import prompting, ratelimit
from app.config import settings
from app.embedding import EmbeddingUnavailable
from app.llm import LLMUnavailable, chat
from app.models import (
    AiEmployee,
    Chunk,
    Document,
    Message,
    MessageRole,
    Session as ChatSession,
    SessionStatus,
    Tenant,
    VisitorMemory,
)
from app.rag import RetrievedChunk, dedupe_hits, doc_ids_for_employee, kb_ids_for_employee, retrieve
from app.utils import j_dump, utcnow
from app.vectorstore.base import VectorStoreUnavailable

logger = logging.getLogger("aics.agent")


@dataclass
class TurnResult:
    reply: str
    status: str
    handoff: bool = False
    handoff_reason: str = ""
    hits: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0
    confidence: int = 0
    degraded: bool = False
    degrade_reason: str = ""
    tokens: int = 0
    intent: dict = field(default_factory=dict)
    model_endpoint: str = ""
    # 消息拆分回复：开启拆分且回答较长时，拆好的分段（reply 仍是完整文本）
    segments: list[str] = field(default_factory=list)
    # 本轮回复是否已写入 messages 表。只有「正常生成」这一条路径会写；
    # 早退分支（无意义输入 / 辱骂 / 广告 / 低置信追问 / 转人工 / 模型不可用）
    # 只写 SYSTEM 说明甚至什么都不写 —— 调用方据此补齐 AI 实际说的话。
    ai_saved: bool = False


def _counter(value: int | None) -> int:
    """把会话上的计数字段读成 int。

    为什么需要它：`clarify_count` / `no_answer_streak` 的默认值是在 **INSERT 时**
    由数据库填的，所以一个**尚未落库**的 Session 对象读出来是 `None`。
    问答测试（Playground）刻意用「不落库的探针 Session」复用本函数，
    于是「无意义输入」和「低置信 + 追问澄清」这两条分支会直接崩在
    `None += 1` / `None < int` 上——而这两条分支恰恰是测试最想覆盖的场景。
    这里不假设对象已落库，`None` 一律当 0。
    """
    return 0 if value is None else int(value)


def _degrade(result: TurnResult, reason: str) -> None:
    """标记本轮降级。

    已有原因时**追加**而不是覆盖：一轮对话可能先因「向量库不可用」降级，
    再因取不到知识而走到「无知识命中」分支。若后者覆盖前者，运维看到
    「无知识命中」会去查知识库，而真正的问题是向量库挂了 —— 排查方向被带偏。
    """
    result.degraded = True
    result.degrade_reason = f"{result.degrade_reason}；{reason}" if result.degrade_reason else reason


async def _overview_fallback_hits(
    db: Session,
    *,
    tenant_id: str,
    kb_ids: list[str],
    doc_ids: list[str],
    query: str,
) -> tuple[list[RetrievedChunk], int]:
    """0 命中时的兜底素材：文档概览块 + 放宽门槛的二次检索。

    场景：主观/口语类提问（"哪块最好用""推荐一款"）与规格表几乎无词面重叠，
    向量+重排可能全军覆没，但知识库其实答得了——概览块里躺着整条产品线的
    型号与关键参数，放宽门槛的二次检索还能捞回最相邻的规格/价格块，
    模型据此才能做有依据的推荐与对比。

    返回 (hits, embed_tokens)；没有任何素材时返回 ([], 0)。
    """
    conds = []
    if doc_ids:
        conds.append(Chunk.doc_id.in_(doc_ids))
    elif kb_ids:
        conds.append(Chunk.kb_id.in_(kb_ids))
    if not conds:
        return [], 0
    rows = db.execute(
        select(Chunk, Document.filename)
        .outerjoin(Document, Document.id == Chunk.doc_id)
        .where(
            Chunk.tenant_id == tenant_id,
            Chunk.enabled.is_(True),
            conds[0],
            or_(Chunk.text.like("【文档概览】%"), Chunk.text.like("【条目速览】%")),
        )
        .order_by(Chunk.doc_id, Chunk.seq)
        .limit(4)
    ).all()
    hits = [
        RetrievedChunk(
            chunk_id=c.id, kb_id=c.kb_id, doc_id=c.doc_id,
            filename=fn or "", score=0.0, text=c.text,
        )
        for c, fn in rows
    ]
    # 二次检索：不带任何分数门槛，直接取与提问最相邻的 3 条（跳过重排，省调用）
    try:
        vhits, tokens = await retrieve(
            db, tenant_id=tenant_id, kb_ids=kb_ids, doc_ids=doc_ids,
            query=query, top_k=3, min_score=0.0, no_threshold=True,
        )
    except Exception:  # noqa: BLE001 - 兜底路径本身不允许把对话流程打断
        logger.warning("概览兜底二次检索失败 tenant=%s", tenant_id, exc_info=True)
        vhits, tokens = [], 0
    seen = {h.chunk_id for h in hits}
    for h in vhits:
        if h.chunk_id not in seen:
            hits.append(h)
    return hits, tokens


def _recent_history(db: Session, tenant_id: str, session_id: str, limit: int = 20) -> list[dict]:
    """当前会话历史。默认 20 条 = 10 轮对话（「记住客户历史」关闭时的默认窗口）。"""
    rows = (
        db.execute(
            select(Message)
            .where(Message.tenant_id == tenant_id, Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


CROSS_SESSION_BUDGET_CHARS = 4000  # ≈2k token，防止跨会话历史把提示词预算吃穿


def _cross_session_history(
    db: Session,
    *,
    tenant_id: str,
    visitor_id: str,
    current_session_id: str,
    max_sessions: int = 3,
) -> list[dict]:
    """「记住客户历史」开启时：拉该访客最近几个**历史会话**的消息拼进上下文。

    只取 visitor/ai 两种角色、按时间从新到旧收集，超出字符预算即停 ——
    保证越近的对话越完整，且总预算可控。每条消息的角色随消息带上，
    模型能看到「客户之前说过什么、我们之前答过什么」。
    """
    sids = db.execute(
        select(ChatSession.id)
        .where(
            ChatSession.tenant_id == tenant_id,
            ChatSession.visitor_id == visitor_id,
            ChatSession.id != current_session_id,
        )
        .order_by(ChatSession.last_active_at.desc())
        .limit(max_sessions)
    ).scalars().all()
    if not sids:
        return []
    rows = (
        db.execute(
            select(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.session_id.in_(sids),
                Message.role.in_((MessageRole.VISITOR, MessageRole.AI)),
                Message.content != "",
            )
            .order_by(Message.created_at.desc())
            .limit(80)
        )
        .scalars()
        .all()
    )
    out: list[dict] = []
    used = 0
    for m in rows:
        content = (m.content or "").strip()
        if not content:
            continue
        if used + len(content) > CROSS_SESSION_BUDGET_CHARS:
            break
        out.append({"role": m.role, "content": content})
        used += len(content)
    return list(reversed(out))


# ---- 自动停止回复：时间窗 ----
_HHMM_RE = None  # 时间格式校验在 schemas 层；这里只做窗口判断


def _tenant_now(tz_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name or "Asia/Shanghai"))
    except Exception:  # noqa: BLE001 - 时区名非法时退回 UTC，不炸对话主链路
        return datetime.now(dt_timezone.utc)


def _in_stop_window(rules_json: str, now_local: datetime) -> bool:
    """当前本地时间是否落在任一停止时段内。

    规则：[{"days":[0..6]，0=周一, "start":"HH:MM", "end":"HH:MM"}]；
    end <= start 视为跨零点（如 22:00-08:00）。组间「或」关系。
    """
    try:
        rules = json.loads(rules_json or "[]")
    except ValueError:
        return False
    if not isinstance(rules, list):
        return False
    weekday = now_local.weekday()  # 0=周一 .. 6=周日
    minutes = now_local.hour * 60 + now_local.minute
    for rule in rules:
        try:
            days = [int(d) for d in (rule.get("days") or [])]
            sh, sm = map(int, str(rule.get("start", "00:00")).split(":"))
            eh, em = map(int, str(rule.get("end", "00:00")).split(":"))
        except (ValueError, AttributeError):
            continue
        if weekday not in days:
            continue
        start, end = sh * 60 + sm, eh * 60 + em
        if start <= end:
            if start <= minutes <= end:
                return True
        else:  # 跨零点
            if minutes >= start or minutes <= end:
                return True
    return False


def _ai_reply_count(db: Session, tenant_id: str, session_id: str) -> int:
    """本会话 AI 已回复的轮次（用于轮次停止条件）。"""
    return int(
        db.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.session_id == session_id,
                Message.role == MessageRole.AI,
            )
        ).scalar_one()
    )


def _visitor_msg_count(db: Session, tenant_id: str, session_id: str) -> int:
    """本会话访客已发送的消息条数（用于消息条数停止条件）。"""
    return int(
        db.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.tenant_id == tenant_id,
                Message.session_id == session_id,
                Message.role == MessageRole.VISITOR,
            )
        ).scalar_one()
    )


def _stop_condition_groups(employee: AiEmployee) -> list[list[dict]]:
    """停止条件组：组内「且」，组间「或」。

    兼容旧数据：条件组为空且旧字段 stop_reply_rounds > 0 时，回退为单组单条件。
    """
    try:
        groups = json.loads(employee.stop_condition_groups or "[]")
    except Exception:  # noqa: BLE001 - 脏数据不炸对话主链路
        groups = []
    if not isinstance(groups, list):
        groups = []
    groups = [g for g in groups if isinstance(g, list) and g]
    if not groups and employee.stop_reply_rounds > 0:
        groups = [[{"type": "ai_rounds", "op": "gte", "value": employee.stop_reply_rounds}]]
    return groups


def _hit_stop_groups(
    db: Session, tenant_id: str, session_id: str, groups: list[list[dict]], ai_count: int
) -> bool:
    """任一条件组的全部条件满足即触发停止。目前仅支持「大于等于」比较。"""
    visitor_count: int | None = None
    for group in groups:
        matched = True
        for cond in group:
            value = int(cond.get("value") or 0)
            if cond.get("type") == "ai_rounds":
                actual = ai_count
            elif cond.get("type") == "visitor_msgs":
                if visitor_count is None:
                    visitor_count = _visitor_msg_count(db, tenant_id, session_id)
                actual = visitor_count
            else:  # 未知条件类型按不满足处理
                matched = False
                break
            if actual < value:
                matched = False
                break
        if matched:
            return True
    return False


def _memory_lines(db: Session, tenant_id: str, visitor_id: str, limit: int = 10) -> list[str]:
    rows = (
        db.execute(
            select(VisitorMemory)
            .where(VisitorMemory.tenant_id == tenant_id, VisitorMemory.visitor_id == visitor_id)
            .order_by(VisitorMemory.updated_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [r.content for r in rows if r.content]


def _add_message(
    db: Session,
    *,
    tenant_id: str,
    session_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
) -> Message:
    msg = Message(
        tenant_id=tenant_id,
        session_id=session_id,
        role=role,
        content=content,
        meta=j_dump(meta or {}),
    )
    db.add(msg)
    return msg


def has_online_agent(db: Session, tenant_id: str) -> bool:
    """本期：租户下存在启用状态的 agent 角色账号即视为有人工可接。

    真正的在线状态（坐席在线/忙碌）由会话接待模块提供，此处留接口。
    """
    from app.models import User, UserRole

    row = db.execute(
        select(User.id)
        .where(
            User.tenant_id == tenant_id,
            User.role == UserRole.AGENT,
            User.status == "active",
        )
        .limit(1)
    ).scalar_one_or_none()
    return row is not None


def _trigger_handoff(session: ChatSession, reason: str) -> None:
    session.status = SessionStatus.QUEUED
    session.handoff_reason = reason[:64]


async def handle_turn(
    db: Session,
    *,
    tenant: Tenant,
    employee: AiEmployee,
    session: ChatSession,
    message: str,
    persist: bool = True,
    origin: str = "chat",
    llm_profile: str = "main",
) -> TurnResult:
    now = utcnow()
    result = TurnResult(reply="", status=session.status)

    # 0. 渠道开关：该渠道未开启 AI 接待 → 直接走人工
    # 本期界面已下线该配置；存量数据里的 channel_switch 默认全放行，判定保留仅作兼容。
    channel_switch = {}
    try:
        channel_switch = json.loads(employee.channel_switch or "{}")
    except ValueError:
        channel_switch = {}
    if channel_switch.get(session.channel) is False:
        _trigger_handoff(session, "该渠道未启用 AI 接待")
        result.reply = prompting.HANDOFF_REPLY
        result.handoff = True
        result.handoff_reason = "该渠道未启用 AI 接待"
        result.status = session.status
        if persist:
            _add_message(db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.SYSTEM,
                         content="渠道未启用 AI 接待，已转入人工队列")
        return result

    # 0.5 自动停止回复（与「未有效回答转人工」并存）。静默分支：不回复、不落消息。
    # 顺序：已停止的会话先挡掉（新会话自动恢复）→ 停止时段 → 停止条件组。
    if session.status == SessionStatus.STOPPED:
        result.status = session.status
        return result
    if employee.stop_time_enabled and _in_stop_window(employee.stop_time_rules, _tenant_now(tenant.timezone)):
        result.status = session.status
        return result
    if employee.stop_reply_enabled:
        # 路由层是「先 add 访客消息、再调 handle_turn」且 autoflush=False，
        # 计数前先 flush，否则当前这条访客消息对 count 查询不可见。
        db.flush()
        groups = _stop_condition_groups(employee)
        if groups and _hit_stop_groups(
            db, tenant.id, session.id, groups, _ai_reply_count(db, tenant.id, session.id)
        ):
            # 条件满足：触发停止。有结束语先发结束语，然后本会话不再回复。
            session.status = SessionStatus.STOPPED
            result.status = session.status
            closing = (employee.stop_reply_message or "").strip()
            if closing:
                result.reply = closing
                if persist:
                    _add_message(
                        db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.AI,
                        content=closing, meta={"stop_reply": True},
                    )
                    result.ai_saved = True
            return result

    # 1. 意图与风险判定
    detected = intent_mod.detect(message)
    result.intent = {
        "handoff": detected.handoff,
        "reason": detected.reason,
        "forced": detected.forced,
        "abuse": detected.abuse,
        "ad": detected.ad,
        "meaningless": detected.meaningless,
        "sub_intents": detected.sub_intents,
        "language": detected.language,
    }
    visitor_language = detected.language or tenant.default_language

    if detected.handoff and detected.forced:
        _trigger_handoff(session, detected.reason)
        online = has_online_agent(db, tenant.id)
        result.reply = prompting.HANDOFF_REPLY if online else prompting.HANDOFF_QUEUED_REPLY
        result.handoff = True
        result.handoff_reason = detected.reason
        result.status = session.status
        if persist:
            _add_message(
                db,
                tenant_id=tenant.id,
                session_id=session.id,
                role=MessageRole.SYSTEM,
                content=f"强制转人工：{detected.reason}",
                meta={"intent": result.intent},
            )
        return result

    if detected.abuse:
        result.reply = "抱歉给您带来不好的体验。请您描述一下具体遇到的问题，我这边帮您处理；也可以为您转接人工客服。"
        result.status = session.status
        return result

    if detected.ad:
        result.reply = "不好意思，这里只受理本业务的咨询。请问您有什么业务问题需要我帮忙？"
        result.status = session.status
        return result

    if detected.meaningless:
        result.reply = prompting.CLARIFY_REPLY
        session.clarify_count = _counter(session.clarify_count) + 1
        result.status = session.status
        return result

    # 2. 检索（知识库绑定优先；未绑库时才用文件级绑定）
    #
    # 为什么库级优先：文件级绑定是**静态快照**——上传新文档后不会自动进入列表，
    # 用户明明"绑定了知识库"，新资料却怎么问都检索不到（实测 score=0）。
    # 库级绑定按 kb_id 过滤，库内新增文档天然纳入，符合"我绑的是这个库"的直觉。
    # 文件级只在没绑库时生效，用于逐个勾选文件的精细控制场景。
    kb_ids = kb_ids_for_employee(db, tenant.id, employee.id)
    doc_ids = [] if kb_ids else doc_ids_for_employee(db, tenant.id, employee.id)
    hits: list[RetrievedChunk] = []
    embed_tokens = 0
    try:
        hits, embed_tokens = await retrieve(
            db, tenant_id=tenant.id, kb_ids=kb_ids, query=message, doc_ids=doc_ids
        )
        hits = dedupe_hits(hits)
    except VectorStoreUnavailable as exc:  # 向量库不可用：降级但不编造
        logger.warning("向量库不可用 tenant=%s err=%s", tenant.id, exc.detail or exc.message)
        _degrade(result, f"向量库不可用：{exc.message}")
    except EmbeddingUnavailable as exc:  # 向量模型不可用
        logger.warning("向量模型不可用 tenant=%s err=%s", tenant.id, exc.detail or exc.message)
        _degrade(result, f"向量模型不可用：{exc.message}")
    except Exception as exc:  # 其它异常：同样不编造答案
        logger.exception("检索失败 tenant=%s", tenant.id)
        _degrade(result, f"检索服务异常：{type(exc).__name__}")

    result.hits = hits
    # score=0 的块是检索层定向注入的兜底素材（如【售后服务】热线块），不代表
    # 检索置信度。置信度与兜底策略判定必须基于真实命中——否则"只剩兜底块"时
    # 模型没机会作答就被固定话术拦截（实测 M038：素材在上下文候选里却被
    # CLARIFY_REPLY 拦截，模型根本没被调用）。
    real_hits = [h for h in hits if h.score > 0]
    result.top_score = real_hits[0].score if real_hits else 0.0
    result.confidence = intent_mod.confidence_from_score(result.top_score)

    # 3. 无命中 / 低置信 → 走配置的兜底策略
    low_confidence = (not real_hits) or result.confidence < employee.confidence_threshold
    if low_confidence:
        policy = employee.low_confidence_policy
        clarify_count = _counter(session.clarify_count)

        # ---- 上下文优先：无知识命中 ≠ 无答案 ----
        # 答案可能就在前几轮对话里（访客提供过的信息、AI 此前回答过的内容）。
        # 有历史时先让模型基于对话记录判断；模型确认答不出来（返回哨兵
        # NO_ANSWER）才继续走追问/兜底/转人工。首轮对话（无历史）直接走原逻辑。
        history_ctx = _recent_history(db, tenant.id, session.id) if persist else []
        if not real_hits and history_ctx:
            ctx_system = (
                "【对话上下文兜底】\n"
                "本轮没有检索到知识片段。请先查看对话记录：\n"
                "- 若答案能从对话记录中得出（访客此前提供过的信息、或你此前回答过的内容），"
                "请直接、如实作答，不得引入对话记录之外的任何业务事实；\n"
                "- 若对话记录中得不出答案，只回复 NO_ANSWER 这一个词，不要有任何其他内容。"
            )
            ctx_messages = prompting.build_messages(ctx_system, history_ctx, message)
            ctx_llm = None
            try:
                ctx_llm = await chat(
                    ctx_messages,
                    temperature=employee.llm_temperature / 100.0,
                    max_tokens=employee.llm_max_tokens,
                    profile=llm_profile,
                )
            except LLMUnavailable as exc:
                logger.warning("上下文兜底调用失败 tenant=%s err=%s", tenant.id, exc.message)
            ratelimit.record_usage(
                db, tenant_id=tenant.id, employee_id=employee.id, kind="llm",
                model=ctx_llm.model if ctx_llm else settings.llm_model or "unknown",
                prompt_tokens=ctx_llm.prompt_tokens if ctx_llm else 0,
                completion_tokens=ctx_llm.completion_tokens if ctx_llm else 0,
                origin=origin, session_id=session.id,
            )
            ctx_reply = (ctx_llm.text or "").strip() if ctx_llm else ""
            if ctx_reply and ctx_reply != "NO_ANSWER":
                result.reply = ctx_reply
                result.tokens = ctx_llm.total_tokens + embed_tokens
                result.model_endpoint = ctx_llm.endpoint
                result.status = session.status
                _degrade(result, "未命中知识片段，基于对话上下文作答")
                session.no_answer_streak = 0
                session.clarify_count = 0
                if embed_tokens:
                    ratelimit.record_usage(
                        db, tenant_id=tenant.id, employee_id=employee.id, kind="embedding",
                        model=settings.embed_model or settings.embed_provider,
                        prompt_tokens=embed_tokens, origin=origin, session_id=session.id,
                    )
                if persist:
                    _add_message(
                        db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.AI,
                        content=result.reply,
                        meta={"context_answer": True, "endpoint": ctx_llm.endpoint,
                              "tokens": result.tokens},
                    )
                    result.ai_saved = True
                session.last_active_at = now
                return result

        # ---- 概览兜底：无知识命中 ≠ 知识库答不了 ----
        # 主观/口语类提问（"哪块最好用""推荐一款"）与规格表几乎无词面重叠，
        # 向量+重排可能 0 命中。注入文档概览块与最相邻资料，让模型做有依据的
        # 推荐/对比；仍以 NO_ANSWER 哨兵防编造，答不了继续走原追问/兜底流程。
        if not real_hits:
            fb_hits, fb_tokens = await _overview_fallback_hits(
                db, tenant_id=tenant.id, kb_ids=kb_ids, doc_ids=doc_ids, query=message
            )
            if fb_hits:
                fb_system = prompting.build_system_prompt(
                    employee=employee,
                    tenant_name=tenant.name,
                    hits=fb_hits,
                    now=now,
                    visitor_language=visitor_language,
                ) + (
                    "\n\n【兜底说明】\n"
                    "本轮常规检索没有命中，以上片段来自知识库的文档概览与最相邻资料。\n"
                    "- 若能据此回答用户问题（如产品推荐、对比、参数查询），请直接、如实作答；\n"
                    "- 若这些资料仍不足以回答，只回复 NO_ANSWER 这一个词，不要有任何其他内容。"
                )
                fb_messages = prompting.build_messages(fb_system, history_ctx, message)
                fb_llm = None
                try:
                    fb_llm = await chat(
                        fb_messages,
                        temperature=employee.llm_temperature / 100.0,
                        max_tokens=employee.llm_max_tokens,
                        profile=llm_profile,
                    )
                except LLMUnavailable as exc:
                    logger.warning("概览兜底调用失败 tenant=%s err=%s", tenant.id, exc.message)
                ratelimit.record_usage(
                    db, tenant_id=tenant.id, employee_id=employee.id, kind="llm",
                    model=fb_llm.model if fb_llm else settings.llm_model or "unknown",
                    prompt_tokens=fb_llm.prompt_tokens if fb_llm else 0,
                    completion_tokens=fb_llm.completion_tokens if fb_llm else 0,
                    origin=origin, session_id=session.id,
                )
                fb_reply = (fb_llm.text or "").strip() if fb_llm else ""
                # degraded（如开发模式 stub 返回「模型未配置」）不算有效回答：
                # 那是降级话术不是答案，必须落回原兜底流程，否则会把降级文本当答案返回。
                if fb_reply and fb_reply != "NO_ANSWER" and fb_llm and not fb_llm.degraded:
                    result.reply = fb_reply
                    result.hits = fb_hits
                    result.tokens = (fb_llm.total_tokens if fb_llm else 0) + embed_tokens + fb_tokens
                    result.model_endpoint = fb_llm.endpoint if fb_llm else ""
                    result.status = session.status
                    _degrade(result, "未命中知识片段，基于文档概览兜底作答")
                    session.no_answer_streak = 0
                    session.clarify_count = 0
                    if embed_tokens or fb_tokens:
                        ratelimit.record_usage(
                            db, tenant_id=tenant.id, employee_id=employee.id, kind="embedding",
                            model=settings.embed_model or settings.embed_provider,
                            prompt_tokens=embed_tokens + fb_tokens, origin=origin,
                        )
                    if persist:
                        _add_message(
                            db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.AI,
                            content=result.reply,
                            meta={"overview_fallback": True,
                                  "endpoint": fb_llm.endpoint if fb_llm else "",
                                  "tokens": result.tokens},
                        )
                        result.ai_saved = True
                    session.last_active_at = now
                    return result

        # 模型基于上下文也答不出（或无历史/模型不可用）→ 原有追问/兜底/转人工流程
        if policy == "clarify" and clarify_count < employee.clarify_rounds:
            session.clarify_count = clarify_count + 1
            result.reply = (
                prompting.CLARIFY_REPLY if real_hits else prompting.NO_ANSWER_REPLY
            )
            result.status = session.status
            _degrade(result, "无知识命中（低置信）")
            if persist:
                _add_message(
                    db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.SYSTEM,
                    content="低置信：追问澄清", meta={"intent": result.intent, "confidence": result.confidence},
                )
            return result

        if policy == "fallback" and hits:
            result.reply = prompting.NO_ANSWER_REPLY
            result.status = session.status
            _degrade(result, "低置信：兜底话术")
            return result

        # clarify 用尽 / handoff 策略
        _trigger_handoff(session, "无知识命中或置信度过低")
        online = has_online_agent(db, tenant.id)
        result.reply = prompting.HANDOFF_REPLY if online else prompting.HANDOFF_QUEUED_REPLY
        result.handoff = True
        result.handoff_reason = session.handoff_reason
        result.status = session.status
        _degrade(result, "低置信转人工")
        if persist:
            _add_message(
                db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.SYSTEM,
                content="低置信转人工", meta={"intent": result.intent, "confidence": result.confidence},
            )
        return result

    # 4. 生成回复
    # 历史窗口：默认当前会话最近 20 条（10 轮）；开启「记住客户历史」后，
    # 额外拼入该客户最近几个历史会话的消息（跨会话连贯服务）。
    history = _recent_history(db, tenant.id, session.id) if persist else []
    if persist and employee.memory_enabled:
        history = _cross_session_history(
            db, tenant_id=tenant.id, visitor_id=session.visitor_id, current_session_id=session.id
        ) + history
    memory = _memory_lines(db, tenant.id, session.visitor_id) if employee.memory_enabled else []
    system_prompt = prompting.build_system_prompt(
        employee=employee,
        tenant_name=tenant.name,
        hits=hits,
        now=now,
        visitor_language=visitor_language,
        memory_lines=memory,
    )
    messages = prompting.build_messages(system_prompt, history, message)

    try:
        llm_result = await chat(
            messages,
            temperature=employee.llm_temperature / 100.0,
            max_tokens=employee.llm_max_tokens,
            profile=llm_profile,
        )
    except LLMUnavailable as exc:
        logger.error("模型不可用 tenant=%s err=%s", tenant.id, exc.detail)
        _trigger_handoff(session, "模型服务不可用")
        result.reply = prompting.DEGRADED_REPLY
        result.handoff = True
        result.handoff_reason = "模型服务不可用"
        result.status = session.status
        _degrade(result, f"模型不可用：{exc.message}")
        if persist:
            _add_message(
                db, tenant_id=tenant.id, session_id=session.id, role=MessageRole.SYSTEM,
                content=f"模型不可用，已转人工：{exc.message}",
            )
        return result

    result.reply = llm_result.text
    result.tokens = llm_result.total_tokens + embed_tokens
    result.model_endpoint = llm_result.endpoint
    result.status = session.status
    # 消息拆分回复：开启且回答较长时拆成多条（reply 仍是完整文本，落库不拆）
    if employee.split_reply_enabled:
        result.segments = prompting.split_reply_segments(result.reply, employee.split_reply_max)
    if llm_result.degraded:
        _degrade(result, f"已切换备用模型端点（{llm_result.endpoint}）")

    # 有效回答 → 清零未回答计数
    session.no_answer_streak = 0
    session.clarify_count = 0

    ratelimit.record_usage(
        db,
        tenant_id=tenant.id,
        employee_id=employee.id,
        kind="llm",
        model=llm_result.model,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        origin=origin,
        session_id=session.id,
    )
    if embed_tokens:
        ratelimit.record_usage(
            db,
            tenant_id=tenant.id,
            employee_id=employee.id,
            kind="embedding",
            model=settings.embed_model or settings.embed_provider,
            prompt_tokens=embed_tokens,
            origin=origin,
            session_id=session.id,
        )

    if persist:
        _add_message(
            db,
            tenant_id=tenant.id,
            session_id=session.id,
            role=MessageRole.AI,
            content=result.reply,
            meta={
                "hits": [
                    {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "kb_id": h.kb_id,
                     "score": round(h.score, 4), "filename": h.filename}
                    for h in hits
                ],
                "confidence": result.confidence,
                "endpoint": llm_result.endpoint,
                "tokens": result.tokens,
                "segments": result.segments,
            },
        )
        result.ai_saved = True
    session.last_active_at = now
    return result


def escalate_no_answer(session: ChatSession, employee: AiEmployee) -> bool:
    """连续未有效回答达阈值 → 停止 AI 回复并转人工（PRD 3.1.5）。"""
    session.no_answer_streak = streak = _counter(session.no_answer_streak) + 1
    if employee.auto_stop_enabled and streak >= employee.auto_stop_threshold:
        _trigger_handoff(session, f"连续 {employee.auto_stop_threshold} 次未有效回答")
        return True
    return False
