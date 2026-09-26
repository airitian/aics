"""审计日志：越权尝试、跨租户查看、导出等必须留痕（PRD 2.5）。"""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditLog
from app.security import Principal
from app.utils import j_dump


def record(
    db: Session,
    *,
    action: str,
    tenant_id: str | None = None,
    actor: Principal | None = None,
    target: str = "",
    result: str = "allow",
    ip: str = "",
    detail: Any = "",
    commit: bool = False,
) -> AuditLog:
    if isinstance(detail, (dict, list)):
        detail = j_dump(detail)
    log = AuditLog(
        tenant_id=tenant_id,
        actor_user_id=actor.user_id if actor else None,
        actor_email=actor.email if actor else "",
        action=action,
        target=target[:190],
        result=result,
        ip=ip or "",
        detail=str(detail or ""),
    )
    db.add(log)
    if commit:
        db.commit()
    return log


def deny(
    db: Session,
    *,
    action: str,
    tenant_id: str | None = None,
    actor: Principal | None = None,
    target: str = "",
    ip: str = "",
    detail: Any = "",
) -> None:
    record(
        db,
        action=action,
        tenant_id=tenant_id,
        actor=actor,
        target=target,
        result="deny",
        ip=ip,
        detail=detail,
        commit=True,
    )
