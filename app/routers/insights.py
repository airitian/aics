"""会话质量洞察与用量看板。

口径纪律（PRD 7.2）：
- 每个展示的指标都要能查到定义；
- 样本不足必须提示，不给出误导性结论；
- 无数据展示空态，**不显示 0%**；
- 未评价不得视为不满意；
- 「解决率」口径未定稿（PRD 待确认项 #2），因此**本接口不输出该指标**，
  只返回口径未定说明，避免数字被误用。
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import ratelimit
from app.config import TEST_CHANNEL, settings
from app.database import get_db
from app.deps import get_scope, get_tenant, require_roles
from app.models import (
    Message,
    MessageRole,
    PlaygroundRun,
    Session as ChatSession,
    SessionStatus,
    Tenant,
    UsageRecord,
    User,
    UserRole,
)
from app.scoping import TenantScope
from app.utils import utcnow

router = APIRouter(prefix="/api/insights", tags=["insights"])
viewer = require_roles(UserRole.TENANT_ADMIN, UserRole.AGENT, UserRole.VIEWER)


def _test_session_ids(scope: TenantScope, tenant_id: str) -> set[str]:
    """「AI 员工测试」产生的会话 id 集合。

    测试会话是真实落库的会话行，所以任何面向「线上质量」的统计都必须显式
    把它们摘出去，否则测试数据会混进运营指标。
    """
    return set(
        scope.db.execute(
            select(ChatSession.id).where(
                ChatSession.tenant_id == tenant_id,
                ChatSession.channel == TEST_CHANNEL,
            )
        )
        .scalars()
        .all()
    )

METRIC_DEFINITIONS = {
    "会话总数": "统计周期内创建的会话数（按会话去重）",
    "AI 独立接待": "周期内未触发转人工的会话数（handoff_reason 为空且当前非排队/人工态）",
    "转人工": "周期内触发过转人工的会话数（同一会话多次转人工只计一次）",
    "拦截率": "AI 独立接待 ÷ 会话总数",
    "转人工率": "转人工 ÷ 会话总数",
    "无命中次数": "AI 判定无相关资料而给出兜底回复的次数（按系统记录计数，非会话去重）",
    "平均首次响应时长": "会话内首条访客消息到首条 AI 回复的平均耗时",
    "样本量": "参与统计的会话数；低于阈值时结论不可作为决策依据",
    "解决率": "**口径待业务方确认（PRD 待确认项 #2），本期不展示**",
}


@router.get("/definition")
def definition() -> dict:
    return {"definitions": METRIC_DEFINITIONS, "min_samples": settings.insight_min_samples}


@router.get("/overview")
def overview(
    days: int = 7,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(viewer),
) -> dict:
    days = max(1, min(90, days))
    since = utcnow() - timedelta(days=days)

    # 排除「AI 员工测试」产生的会话：它们是与线上同构的真实会话行（这样才有
    # 多轮上下文），不排除就会把会话总数、AI 独立接待、拦截率一起算进去，
    # 运营看到的数字就成了假的。
    sessions = [
        s
        for s in scope.list(ChatSession)
        if s.created_at and s.created_at >= since and s.channel != TEST_CHANNEL
    ]
    total = len(sessions)

    if total == 0:
        return {
            "empty": True,
            "period_days": days,
            "message": "当前时间范围内还没有会话数据",
            "definitions": METRIC_DEFINITIONS,
        }

    handoff_sessions = [s for s in sessions if (s.handoff_reason or "").strip()]
    ai_only = [s for s in sessions if not (s.handoff_reason or "").strip()]
    session_ids = [s.id for s in sessions]

    no_hit = int(
        scope.db.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.tenant_id == tenant.id,
                Message.session_id.in_(session_ids),
                Message.role == MessageRole.SYSTEM,
                Message.content.like("%低置信%"),
            )
        ).scalar_one()
    )
    ai_replies = int(
        scope.db.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.tenant_id == tenant.id,
                Message.session_id.in_(session_ids),
                Message.role == MessageRole.AI,
            )
        ).scalar_one()
    )

    # 平均首次响应时长
    rows = scope.db.execute(
        select(Message.session_id, Message.role, Message.created_at)
        .where(Message.tenant_id == tenant.id, Message.session_id.in_(session_ids))
        .order_by(Message.session_id, Message.created_at)
    ).all()
    first_visitor: dict[str, object] = {}
    deltas: list[float] = []
    for sid, role, created in rows:
        if role == MessageRole.VISITOR and sid not in first_visitor:
            first_visitor[sid] = created
        elif role in (MessageRole.AI, MessageRole.AGENT) and sid in first_visitor and created:
            base = first_visitor.pop(sid)
            deltas.append((created - base).total_seconds() * 1000)
    avg_first_ms = int(sum(deltas) / len(deltas)) if deltas else 0

    status_counts: dict[str, int] = {}
    for s in sessions:
        status_counts[s.status] = status_counts.get(s.status, 0) + 1

    enough = total >= settings.insight_min_samples
    return {
        "empty": False,
        "period_days": days,
        "sample_size": total,
        "sample_enough": enough,
        "sample_hint": "" if enough else f"样本量 {total} 条，少于 {settings.insight_min_samples} 条，结论不可作为决策依据",
        "data_updated_at": utcnow().isoformat(),
        "metrics": {
            "sessions_total": total,
            "sessions_ai_only": len(ai_only),
            "sessions_handoff": len(handoff_sessions),
            "intercept_rate": round(len(ai_only) / total * 100, 1),
            "handoff_rate": round(len(handoff_sessions) / total * 100, 1),
            "no_hit_count": no_hit,
            "no_hit_ratio": round(no_hit / ai_replies * 100, 1) if ai_replies else None,
            "ai_replies": ai_replies,
            "avg_first_response_ms": avg_first_ms,
            "status_counts": status_counts,
            "resolve_rate": None,
        },
        "definitions": METRIC_DEFINITIONS,
    }


@router.get("/unresolved")
def unresolved(
    days: int = 7,
    limit: int = 50,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(viewer),
) -> dict:
    """无命中 / 低置信问题清单：支持一键跳转知识库补充。"""
    days = max(1, min(90, days))
    since = utcnow() - timedelta(days=days)
    stmt = select(Message).where(
        Message.tenant_id == tenant.id,
        Message.role == MessageRole.SYSTEM,
        Message.content.like("%低置信%"),
        Message.created_at >= since,
    )
    # 在 SQL 层就排除测试会话：先 limit 再在 Python 里过滤会让返回条数莫名变少
    test_ids = _test_session_ids(scope, tenant.id)
    if test_ids:
        stmt = stmt.where(Message.session_id.notin_(test_ids))
    rows = (
        scope.db.execute(stmt.order_by(Message.created_at.desc()).limit(min(200, max(1, limit))))
        .scalars()
        .all()
    )

    session_ids = list({r.session_id for r in rows})
    visitors: dict[str, str] = {}
    if session_ids:
        for s in scope.db.execute(
            select(ChatSession).where(
                ChatSession.tenant_id == tenant.id, ChatSession.id.in_(session_ids)
            )
        ).scalars().all():
            visitors[s.id] = s.visitor_id

    items = []
    for row in rows:
        last_question = scope.db.execute(
            select(Message.content)
            .where(
                Message.tenant_id == tenant.id,
                Message.session_id == row.session_id,
                Message.role == MessageRole.VISITOR,
                Message.created_at <= row.created_at,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        items.append(
            {
                "session_id": row.session_id,
                "visitor_id": visitors.get(row.session_id, ""),
                "question": last_question or "",
                "note": row.content,
                "at": row.created_at.isoformat(),
            }
        )
    return {"items": items, "total": len(items), "period_days": days}


@router.get("/usage")
def usage(
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(viewer),
) -> dict:
    rows = scope.db.execute(
        select(
            UsageRecord.kind,
            func.sum(UsageRecord.prompt_tokens),
            func.sum(UsageRecord.completion_tokens),
            func.sum(UsageRecord.total_tokens),
            func.count(),
        )
        .where(UsageRecord.tenant_id == tenant.id)
        .group_by(UsageRecord.kind)
    ).all()
    by_kind = [
        {
            "kind": r[0],
            "prompt_tokens": int(r[1] or 0),
            "completion_tokens": int(r[2] or 0),
            "total_tokens": int(r[3] or 0),
            "calls": int(r[4] or 0),
        }
        for r in rows
    ]

    by_employee = []
    emp_rows = scope.db.execute(
        select(
            UsageRecord.employee_id,
            func.sum(UsageRecord.total_tokens),
            func.count(),
        )
        .where(UsageRecord.tenant_id == tenant.id, UsageRecord.employee_id.is_not(None))
        .group_by(UsageRecord.employee_id)
    ).all()
    for eid, tokens, calls in emp_rows:
        by_employee.append(
            {"employee_id": eid, "total_tokens": int(tokens or 0), "calls": int(calls or 0)}
        )

    playground_calls = int(
        scope.db.execute(
            select(func.count()).select_from(PlaygroundRun).where(PlaygroundRun.tenant_id == tenant.id)
        ).scalar_one()
    )

    tokens_today = ratelimit.tokens_used_today(scope.db, tenant.id)
    return {
        "tenant_id": tenant.id,
        "plan": tenant.plan,
        "daily_token_quota": tenant.daily_token_quota,
        "tokens_today": tokens_today,
        "quota_used_percent": round(tokens_today / tenant.daily_token_quota * 100, 2)
        if tenant.daily_token_quota
        else None,
        "rpm_limit": tenant.rpm_limit,
        "by_kind": by_kind,
        "by_employee": by_employee,
        "playground_runs": playground_calls,
        "playground_keep_days": settings.playground_keep_days,
    }


@router.get("/audit")
def audit_trail(
    limit: int = 100,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(require_roles(UserRole.TENANT_ADMIN)),
) -> dict:
    """本租户的审计记录（越权尝试、配置变更等）。"""
    from app.models import AuditLog

    rows = scope.db.execute(
        select(AuditLog)
        .where(AuditLog.tenant_id == scope.tenant_id)
        .order_by(AuditLog.created_at.desc())
        .limit(min(500, max(1, limit)))
    ).scalars().all()
    return {
        "items": [
            {
                "id": r.id,
                "action": r.action,
                "target": r.target,
                "result": r.result,
                "actor_email": r.actor_email,
                "ip": r.ip,
                "detail": r.detail,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
        "keep_days": settings.audit_keep_days,
    }
