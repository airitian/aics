"""AI 员工测试：聊天式真实对话。

与「问答测试」(playground) 的分工：
- playground：单轮、`persist=False` 的探针会话，服务于批量回归评测；
- 本模块：**多轮真实链路**。消息与回复都落库，直接复用与线上完全相同的
  `agent.handle_turn`，因此多轮上下文、兜底策略、转人工判定、记忆与配额
  计量都与线上一致 —— 这里测出来的行为就是访客会遇到的行为。

隔离保证（"测试不影响线上"这句话要有代码兜着）：
- 会话 `channel` 固定为 `TEST_CHANNEL`，`visitor_id` 固定为 `tester:<user_id>`；
- 概览统计按 channel 排除测试会话，不污染线上指标；
- 会话归属做**三重校验**（租户作用域 + 员工 + 测试者），拿别人的 id 也读不到。

一个容易踩的坑：`SessionLocal(autoflush=False)` 意味着 `db.add(...)` 之后
不会自动 flush。所以「先落访客消息、再调 handle_turn」不会让当前这条消息
进入 `_recent_history`，从而与 `build_messages` 末尾追加的 user 消息重复。
**这里不能为了拿 id 而手动 flush**，否则上下文里就会出现两条一样的问题。
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, func, select

from app import agent as agent_mod
from app import prompting
from app.config import TEST_CHANNEL
from app.deps import get_scope, get_tenant, require_roles
from app.models import (
    AiEmployee,
    Message,
    MessageRole,
    Session as ChatSession,
    SessionStatus,
    Tenant,
    User,
    UserRole,
)
from app.schemas import HitOut, TestChatIn, TestChatOut
from app.scoping import TenantScope
from app.utils import new_id, utcnow

logger = logging.getLogger("aics.testchat")

router = APIRouter(prefix="/api/testchat", tags=["testchat"])
tester = require_roles(UserRole.TENANT_ADMIN, UserRole.CONFIG_EDITOR)

# LLM 档位路由：网页端「点击测试」不带该头 → 固定走主档（DeepSeek 官方 key）；
# 自动化开发测试带 X-LLM-Profile: dev → 走 dev 档（旧中转 key，见 config.llm_dev_*）。
# 头里传别的值一律按主档处理，防止误传把生产流量带偏。
DEV_PROFILE_HEADER = "x-llm-profile"
VALID_PROFILES = {"main", "dev"}


def _llm_profile(request: Request) -> str:
    value = (request.headers.get(DEV_PROFILE_HEADER) or "").strip().lower()
    return value if value in VALID_PROFILES else "main"


def _visitor_id(user: User) -> str:
    """测试者自己的访客标识。

    用 user.id 而不是固定值：多个测试者同时测同一个员工时，各自的上下文
    必须互不可见，否则 A 的前几轮对话会出现在 B 的上下文里。
    """
    return f"tester:{user.id}"


def _employee(scope: TenantScope, employee_id: str) -> AiEmployee:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    return employee


def _owned_session(
    scope: TenantScope, employee: AiEmployee, session_id: str, user: User
) -> ChatSession:
    """取出属于「当前测试者 + 当前员工」的测试会话。

    校验缺一不可：只按 id 取会把别人的测试会话（甚至线上访客会话）当测试用，
    上下文串台且破坏隔离。
    """
    row = scope.get(ChatSession, session_id)
    if (
        row is None
        or row.employee_id != employee.id
        or row.channel != TEST_CHANNEL
        or row.visitor_id != _visitor_id(user)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="测试会话不存在")
    return row


def _session_out(row: ChatSession, msg_count: int = 0) -> dict:
    return {
        "id": row.id,
        "employee_id": row.employee_id,
        "status": row.status,
        "handoff_reason": row.handoff_reason,
        "message_count": msg_count,
        "last_active_at": row.last_active_at.isoformat() if row.last_active_at else "",
        "created_at": row.created_at.isoformat() if row.created_at else "",
    }


def _message_out(row: Message) -> dict:
    return {
        "id": row.id,
        "role": row.role,
        "content": row.content,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "meta": row.meta_data,
    }


@router.post("/{employee_id}/sessions")
def create_session(
    employee_id: str,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = _employee(scope, employee_id)
    row = ChatSession(
        id=new_id(),
        tenant_id=tenant.id,
        employee_id=employee.id,
        visitor_id=_visitor_id(user),
        channel=TEST_CHANNEL,
        status=SessionStatus.AI,
    )
    scope.db.add(row)
    scope.db.commit()
    return _session_out(row)


@router.get("/{employee_id}/sessions")
def list_sessions(
    employee_id: str,
    limit: int = 30,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = _employee(scope, employee_id)
    rows = (
        scope.db.execute(
            select(ChatSession)
            .where(
                ChatSession.tenant_id == employee.tenant_id,
                ChatSession.employee_id == employee.id,
                ChatSession.channel == TEST_CHANNEL,
                ChatSession.visitor_id == _visitor_id(user),
            )
            .order_by(ChatSession.last_active_at.desc())
            .limit(min(100, max(1, limit)))
        )
        .scalars()
        .all()
    )
    ids = [r.id for r in rows]
    counts: dict[str, int] = {}
    if ids:
        counts = dict(
            scope.db.execute(
                select(Message.session_id, func.count())
                .where(Message.tenant_id == employee.tenant_id, Message.session_id.in_(ids))
                .group_by(Message.session_id)
            ).all()
        )
    return {"items": [_session_out(r, int(counts.get(r.id, 0))) for r in rows]}


@router.get("/{employee_id}/sessions/{session_id}/messages")
def messages(
    employee_id: str,
    session_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = _employee(scope, employee_id)
    row = _owned_session(scope, employee, session_id, user)
    rows = (
        scope.db.execute(
            select(Message)
            .where(Message.tenant_id == row.tenant_id, Message.session_id == row.id)
            .order_by(Message.created_at)
        )
        .scalars()
        .all()
    )
    return {"session": _session_out(row, len(rows)), "items": [_message_out(m) for m in rows]}


@router.post("/{employee_id}/sessions/{session_id}/messages", response_model=TestChatOut)
async def send_message(
    employee_id: str,
    session_id: str,
    payload: TestChatIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> TestChatOut:
    employee = _employee(scope, employee_id)
    row = _owned_session(scope, employee, session_id, user)
    db = scope.db

    # 访客消息先落库（与线上 chat 一致：消息不能丢）。
    # autoflush=False，故此处不 flush —— 详见模块文档字符串里的说明。
    db.add(
        Message(
            tenant_id=tenant.id,
            session_id=row.id,
            role=MessageRole.VISITOR,
            content=payload.message,
            meta="{}",
        )
    )
    row.last_active_at = utcnow()

    started = time.perf_counter()
    result = await agent_mod.handle_turn(
        db,
        tenant=tenant,
        employee=employee,
        session=row,
        message=payload.message,
        persist=True,
        origin="playground",
        llm_profile=_llm_profile(request),
    )

    # 与线上链路保持同一套「连续未有效回答 → 停止 AI 回复」逻辑。
    # 漏掉这段，测试里就永远测不出线上会发生的兜底升级，测试会骗人。
    if result.degraded and not result.handoff and "低置信" in (result.degrade_reason or ""):
        if agent_mod.escalate_no_answer(row, employee):
            result.handoff = True
            result.handoff_reason = row.handoff_reason
            result.reply = (
                prompting.HANDOFF_REPLY
                if agent_mod.has_online_agent(db, tenant.id)
                else prompting.HANDOFF_QUEUED_REPLY
            )

    # 补齐「AI 实际说的话」：早退分支（无意义输入 / 低置信追问 / 转人工 /
    # 模型不可用）只写了 SYSTEM 说明。测试页靠历史回放展示对话，
    # 不补的话刷新一下就发现 AI 的兜底回复消失了 —— 测试者会以为它没回过。
    if result.reply and not result.ai_saved:
        db.add(
            Message(
                tenant_id=tenant.id,
                session_id=row.id,
                role=MessageRole.AI,
                content=result.reply,
            )
        )
        result.ai_saved = True

    latency_ms = int((time.perf_counter() - started) * 1000)
    db.commit()

    return TestChatOut(
        session_id=row.id,
        reply=result.reply,
        status=result.status,
        handoff=result.handoff,
        handoff_reason=result.handoff_reason,
        hits=[
            HitOut(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                kb_id=h.kb_id,
                filename=h.filename,
                score=round(h.score, 4),
                text=h.text,
            )
            for h in result.hits
        ],
        top_score=round(result.top_score, 4),
        confidence=result.confidence,
        latency_ms=latency_ms,
        tokens=result.tokens,
        degraded=result.degraded,
        degrade_reason=result.degrade_reason,
    )


@router.delete("/{employee_id}/sessions/{session_id}")
def delete_session(
    employee_id: str,
    session_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = _employee(scope, employee_id)
    row = _owned_session(scope, employee, session_id, user)
    # 显式删消息，不依赖库层 CASCADE：换库（SQLite → PostgreSQL）时
    # 外键级联的默认行为并不一致，显式删才两边都稳。
    scope.db.execute(delete(Message).where(Message.session_id == row.id))
    scope.db.delete(row)
    scope.db.commit()
    return {"ok": True}


@router.delete("/{employee_id}/sessions")
def clear_sessions(
    employee_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    """清空当前测试者的全部测试会话。"""
    employee = _employee(scope, employee_id)
    rows = (
        scope.db.execute(
            select(ChatSession).where(
                ChatSession.tenant_id == employee.tenant_id,
                ChatSession.employee_id == employee.id,
                ChatSession.channel == TEST_CHANNEL,
                ChatSession.visitor_id == _visitor_id(user),
            )
        )
        .scalars()
        .all()
    )
    ids = [r.id for r in rows]
    if ids:
        scope.db.execute(delete(Message).where(Message.session_id.in_(ids)))
        for r in rows:
            scope.db.delete(r)
    scope.db.commit()
    return {"ok": True, "deleted": len(ids)}
