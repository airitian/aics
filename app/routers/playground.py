"""问答测试（Playground）：不影响线上会话，也不产生工单。

实现方式：复用对话编排，但 persist=False —— 消息不落 messages 表、
不创建 session 行，只记录测试结果与用量（origin=playground）。
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import agent as agent_mod
from app.config import settings
from app.database import get_db
from app.deps import get_scope, get_tenant, require_roles
from app.models import AiEmployee, PlaygroundRun, Session as ChatSession, SessionStatus, Tenant, User, UserRole
from app.schemas import BatchTestIn, PlaygroundIn, PlaygroundOut
from app.scoping import TenantScope
from app.utils import j_dump, j_load, new_id, utcnow

logger = logging.getLogger("aics.playground")

router = APIRouter(prefix="/api/playground", tags=["playground"])
tester = require_roles(UserRole.TENANT_ADMIN, UserRole.CONFIG_EDITOR)


def _run_out(row: PlaygroundRun) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "answer": row.answer,
        "hits": j_load(row.hits, []),
        "score": row.score,
        "latency_ms": row.latency_ms,
        "tokens": row.tokens,
        "ok": row.ok,
        "created_at": row.created_at.isoformat(),
    }


async def _ask_once(
    db: Session,
    *,
    tenant: Tenant,
    employee: AiEmployee,
    user: User,
    question: str,
) -> dict:
    probe = ChatSession(
        id=new_id(),
        tenant_id=tenant.id,
        employee_id=employee.id,
        visitor_id=f"playground:{user.id}",
        channel="playground",
        status=SessionStatus.AI,
    )
    started = time.perf_counter()
    try:
        result = await agent_mod.handle_turn(
            db,
            tenant=tenant,
            employee=employee,
            session=probe,          # 不 add 到 session，因此不会落库
            message=question,
            persist=False,
            origin="playground",
        )
        latency = int((time.perf_counter() - started) * 1000)
        hits = [
            {
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "kb_id": h.kb_id,
                "filename": h.filename,
                "score": round(h.score, 4),
                "text": h.text[:400],
            }
            for h in result.hits
        ]
        db.add(
            PlaygroundRun(
                id=new_id(),
                tenant_id=tenant.id,
                employee_id=employee.id,
                question=question[:2000],
                answer=result.reply,
                hits=j_dump(hits),
                score=int(result.top_score * 100),
                latency_ms=latency,
                tokens=result.tokens,
                ok=True,
            )
        )
        return {
            "question": question,
            "answer": result.reply,
            "hits": hits,
            "top_score": result.top_score,
            "latency_ms": latency,
            "tokens": result.tokens,
            "ok": True,
            "degraded": result.degraded,
            "degrade_reason": result.degrade_reason,
            "handoff": result.handoff,
        }
    except Exception as exc:  # noqa: BLE001 - 单条失败不影响整批
        logger.exception("问答测试失败")
        latency = int((time.perf_counter() - started) * 1000)
        db.add(
            PlaygroundRun(
                id=new_id(),
                tenant_id=tenant.id,
                employee_id=employee.id,
                question=question[:2000],
                answer="",
                hits="[]",
                latency_ms=latency,
                ok=False,
            )
        )
        return {
            "question": question,
            "answer": "",
            "hits": [],
            "top_score": 0.0,
            "latency_ms": latency,
            "tokens": 0,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@router.post("/{employee_id}/ask", response_model=PlaygroundOut)
async def ask(
    employee_id: str,
    payload: PlaygroundIn,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> PlaygroundOut:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    data = await _ask_once(scope.db, tenant=tenant, employee=employee, user=user, question=payload.question)
    scope.db.commit()
    return PlaygroundOut(
        question=data["question"],
        answer=data.get("answer", ""),
        hits=data.get("hits", []),
        top_score=data.get("top_score", 0.0),
        latency_ms=data.get("latency_ms", 0),
        tokens=data.get("tokens", 0),
        ok=data.get("ok", False),
        error=data.get("error", ""),
    )


@router.post("/{employee_id}/batch")
async def batch(
    employee_id: str,
    payload: BatchTestIn,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")

    questions = [q.strip() for q in payload.questions if q and q.strip()]
    if not questions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="问题列表为空")
    truncated = 0
    normalized: list[str] = []
    for q in questions:
        if len(q) > 2000:
            q = q[:2000]
            truncated += 1
        normalized.append(q)

    taken = normalized[: settings.batch_test_max]
    rejected = normalized[settings.batch_test_max :]

    items = []
    for q in taken:
        items.append(await _ask_once(scope.db, tenant=tenant, employee=employee, user=user, question=q))
    scope.db.commit()

    return {
        "items": items,
        "executed": len(taken),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "truncated": truncated,
        "max_batch": settings.batch_test_max,
    }


@router.get("/{employee_id}/runs")
def runs(
    employee_id: str,
    limit: int = 50,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    rows = scope.db.execute(
        select(PlaygroundRun)
        .where(
            PlaygroundRun.tenant_id == employee.tenant_id,
            PlaygroundRun.employee_id == employee_id,
        )
        .order_by(PlaygroundRun.created_at.desc())
        .limit(min(200, max(1, limit)))
    ).scalars().all()
    return {"items": [_run_out(r) for r in rows]}


@router.delete("/{employee_id}/runs")
def clear_runs(
    employee_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(tester),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    rows = scope.db.execute(
        select(PlaygroundRun).where(
            PlaygroundRun.tenant_id == employee.tenant_id,
            PlaygroundRun.employee_id == employee_id,
        )
    ).scalars().all()
    for row in rows:
        scope.db.delete(row)
    scope.db.commit()
    return {"ok": True, "deleted": len(rows), "keep_days": settings.playground_keep_days}
