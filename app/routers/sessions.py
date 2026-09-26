"""人工兜底与协作：坐席侧会话接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.config import TEST_CHANNEL
from app.database import get_db
from app.deps import get_scope, get_tenant, require_roles
from app.docstore import read_chat_image
from app.models import (
    ChatImage,
    Message,
    MessageRole,
    Session as ChatSession,
    SessionStatus,
    Tenant,
    User,
    UserRole,
)
from app.ratelimit import record_usage  # noqa: F401  (保持导入一致，便于后续扩展)
from app.schemas import AgentReplyIn
from app.scoping import TenantScope
from app.utils import utcnow

router = APIRouter(prefix="/api", tags=["sessions"])
agent_role = require_roles(UserRole.TENANT_ADMIN, UserRole.AGENT, UserRole.CONFIG_EDITOR)


def _session_out(row: ChatSession) -> dict:
    return {
        "id": row.id,
        "employee_id": row.employee_id,
        "visitor_id": row.visitor_id,
        "channel": row.channel,
        "status": row.status,
        "handoff_reason": row.handoff_reason,
        "assigned_user_id": row.assigned_user_id,
        "no_answer_streak": row.no_answer_streak,
        "last_active_at": row.last_active_at.isoformat(),
        "created_at": row.created_at.isoformat(),
    }


@router.get("/sessions")
def list_sessions(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = 50,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    # 坐席侧只应看到线上会话：「AI 员工测试」产生的会话不代表真实接待，
    # 混进来会让坐席以为有待处理来访（测试数据不该驱动排班与响应）。
    rows = [r for r in scope.list(ChatSession) if r.channel != TEST_CHANNEL]
    if status_filter:
        rows = [r for r in rows if r.status == status_filter]
    rows.sort(key=lambda r: r.last_active_at, reverse=True)
    return {"items": [_session_out(r) for r in rows[: min(200, max(1, limit))]]}


@router.get("/sessions/queue")
def queue(
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    rows = [
        r
        for r in scope.list(ChatSession)
        if r.status == SessionStatus.QUEUED and r.channel != TEST_CHANNEL
    ]
    rows.sort(key=lambda r: r.last_active_at)
    return {"items": [_session_out(r) for r in rows], "total": len(rows)}


@router.get("/sessions/{session_id}/messages")
def messages(
    session_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    rows = scope.db.execute(
        select(Message)
        .where(Message.tenant_id == session.tenant_id, Message.session_id == session_id)
        .order_by(Message.created_at)
    ).scalars().all()
    return {
        "session": _session_out(session),
        "items": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "meta": m.meta_data,
                "images": [
                    {
                        "id": i["id"],
                        "url": f"/api/sessions/{session_id}/images/{i['id']}",
                        "ocr_status": i.get("ocr_status", ""),
                    }
                    for i in (m.meta_data.get("images") or [])
                    if i.get("id")
                ],
            }
            for m in rows
        ],
    }


@router.get("/sessions/{session_id}/images/{image_id}")
def session_image(
    session_id: str,
    image_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> Response:
    """坐席取看会话里客户发的原图。

    人工接管时坐席必须能看到客户拍的是什么 —— 转写文本会丢掉视觉细节
    （划痕位置、装配方向），这些恰恰是售后沟通里最要紧的信息。
    """
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    row = scope.db.execute(
        select(ChatImage).where(
            ChatImage.id == image_id,
            ChatImage.tenant_id == session.tenant_id,
            ChatImage.session_id == session_id,
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片不存在")
    data = read_chat_image(row.rel_path)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片文件已丢失")
    return Response(content=data, media_type=row.mime or "image/png")


@router.post("/sessions/{session_id}/takeover")
def takeover(
    session_id: str,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    session.status = SessionStatus.HUMAN
    session.assigned_user_id = user.id
    # 接管即静默 AI：chat 接口见 status=HUMAN 时不再生成回复，
    # 因此不存在「AI 与人工同时说话」的窗口（PRD 6.2）。
    scope.db.add(
        Message(
            tenant_id=tenant.id,
            session_id=session.id,
            role=MessageRole.SYSTEM,
            content=f"人工客服 {user.name or user.email} 已接管，AI 暂停回复",
        )
    )
    session.last_active_at = utcnow()
    scope.db.commit()
    audit.record(scope.db, action="session.takeover", tenant_id=tenant.id, target=session_id,
                 detail={"agent": user.email}, commit=True)
    return {"ok": True, "status": session.status}


@router.post("/sessions/{session_id}/reply")
def reply(
    session_id: str,
    payload: AgentReplyIn,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    if session.status != SessionStatus.HUMAN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请先接管会话再由人工回复，避免 AI 与人工同时发言",
        )
    msg = Message(
        tenant_id=tenant.id,
        session_id=session.id,
        role=MessageRole.AGENT,
        content=payload.content,
    )
    scope.db.add(msg)
    session.last_active_at = utcnow()
    scope.db.commit()
    return {"ok": True, "message_id": msg.id}


@router.post("/sessions/{session_id}/release")
def release(
    session_id: str,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    session.status = SessionStatus.AI
    session.assigned_user_id = None
    session.handoff_reason = ""
    session.no_answer_streak = 0
    session.clarify_count = 0
    scope.db.add(
        Message(
            tenant_id=tenant.id,
            session_id=session.id,
            role=MessageRole.SYSTEM,
            content="会话已交回 AI 接待",
        )
    )
    scope.db.commit()
    return {"ok": True, "status": session.status}


@router.post("/sessions/{session_id}/close")
def close(
    session_id: str,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(agent_role),
) -> dict:
    session = scope.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    session.status = SessionStatus.CLOSED
    scope.db.commit()
    return {"ok": True, "status": session.status}
