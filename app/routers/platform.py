"""平台运营路由（跨租户）。

平台账号是唯一允许跨租户的角色（PRD 2.5：平台运维跨租户查看需独立权限并留痕），
本模块所有跨租户动作都会写审计。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit, ratelimit
from app.database import get_db
from app.deps import client_ip, require_roles
from app.models import (
    AiEmployee,
    AuditLog,
    Chunk,
    Document,
    Message,
    Session as ChatSession,
    Tenant,
    TenantStatus,
    UsageRecord,
    User,
    UserRole,
)
from app.provision import create_tenant
from app.schemas import TenantCreateIn, TenantOut, TenantUpdateIn
from app.teardown import purge_tenant_data
from app.utils import new_id
from app.vectorstore import build_vector_store

router = APIRouter(prefix="/api/platform", tags=["platform"])
platform_only = require_roles()  # 仅平台管理员


def _audit(db, request, action, target, result="allow", detail="", tenant_id=None):
    audit.record(
        db,
        action=action,
        tenant_id=tenant_id,
        target=target,
        result=result,
        ip=client_ip(request),
        detail=detail,
        commit=True,
    )


@router.post("/tenants", response_model=TenantOut)
def create(
    payload: TenantCreateIn,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(platform_only),
) -> TenantOut:
    exists = db.execute(select(User.id).where(User.email == payload.admin_email.strip().lower())).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该邮箱已被使用")
    tenant, _ = create_tenant(
        db,
        name=payload.name,
        admin_email=payload.admin_email,
        admin_password=payload.admin_password,
        plan=payload.plan,
        daily_token_quota=payload.daily_token_quota,
        rpm_limit=payload.rpm_limit,
        timezone=payload.timezone,
        default_language=payload.default_language,
    )
    _audit(db, request, "platform.tenant.create", tenant.id, tenant_id=tenant.id,
           detail={"name": tenant.name, "plan": tenant.plan})
    return TenantOut(**{k: getattr(tenant, k) for k in TenantOut.model_fields if hasattr(tenant, k)})


@router.get("/tenants")
def list_tenants(
    db: Session = Depends(get_db),
    user: User = Depends(platform_only),
) -> dict:
    tenants = db.execute(select(Tenant).order_by(Tenant.created_at.desc())).scalars().all()
    items = []
    for t in tenants:
        items.append(
            {
                "id": t.id,
                "name": t.name,
                "slug": t.slug,
                "status": t.status,
                "plan": t.plan,
                "daily_token_quota": t.daily_token_quota,
                "rpm_limit": t.rpm_limit,
                "timezone": t.timezone,
                "created_at": t.created_at.isoformat(),
                "employees": int(
                    db.execute(
                        select(func.count()).select_from(AiEmployee).where(AiEmployee.tenant_id == t.id)
                    ).scalar_one()
                ),
                "sessions": int(
                    db.execute(
                        select(func.count()).select_from(ChatSession).where(ChatSession.tenant_id == t.id)
                    ).scalar_one()
                ),
                "tokens_today": ratelimit.tokens_used_today(db, t.id),
            }
        )
    return {"items": items, "total": len(items)}


@router.patch("/tenants/{tenant_id}", response_model=TenantOut)
def update_tenant(
    tenant_id: str,
    payload: TenantUpdateIn,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(platform_only),
) -> TenantOut:
    tenant = db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="租户不存在")
    changes = payload.model_dump(exclude_unset=True)
    for key, value in changes.items():
        setattr(tenant, key, value)
    _audit(db, request, "platform.tenant.update", tenant.id, tenant_id=tenant.id, detail=changes)
    db.commit()
    return TenantOut(**{k: getattr(tenant, k) for k in TenantOut.model_fields if hasattr(tenant, k)})


@router.get("/tenants/{tenant_id}/usage")
def tenant_usage(
    tenant_id: str,
    days: int = 7,
    db: Session = Depends(get_db),
    user: User = Depends(platform_only),
) -> dict:
    tenant = db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="租户不存在")
    rows = db.execute(
        select(
            UsageRecord.kind,
            func.sum(UsageRecord.prompt_tokens),
            func.sum(UsageRecord.completion_tokens),
            func.sum(UsageRecord.total_tokens),
            func.count(),
        )
        .where(UsageRecord.tenant_id == tenant_id)
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
    return {
        "tenant_id": tenant_id,
        "tenant_name": tenant.name,
        "daily_token_quota": tenant.daily_token_quota,
        "tokens_today": ratelimit.tokens_used_today(db, tenant_id),
        "by_kind": by_kind,
    }


@router.get("/overview")
def overview(db: Session = Depends(get_db), user: User = Depends(platform_only)) -> dict:
    tenants = db.execute(select(Tenant)).scalars().all()
    total_tokens_today = sum(ratelimit.tokens_used_today(db, t.id) for t in tenants)
    return {
        "tenants": len(tenants),
        "active_tenants": sum(1 for t in tenants if t.status == TenantStatus.ACTIVE),
        "employees": int(db.execute(select(func.count()).select_from(AiEmployee)).scalar_one()),
        "sessions": int(db.execute(select(func.count()).select_from(ChatSession)).scalar_one()),
        "messages": int(db.execute(select(func.count()).select_from(Message)).scalar_one()),
        "documents": int(db.execute(select(func.count()).select_from(Document)).scalar_one()),
        "chunks": int(db.execute(select(func.count()).select_from(Chunk)).scalar_one()),
        "tokens_today": total_tokens_today,
        "denied_audits": int(
            db.execute(
                select(func.count()).select_from(AuditLog).where(AuditLog.result == "deny")
            ).scalar_one()
        ),
    }


@router.post("/tenants/{tenant_id}/purge")
def purge_tenant(
    tenant_id: str,
    request: Request,
    confirm: str = "",
    db: Session = Depends(get_db),
    user: User = Depends(platform_only),
) -> dict:
    """租户注销：彻底删除该租户全部数据（PRD 2.5）。"""
    tenant = db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="租户不存在")
    if confirm != tenant.slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"需要 confirm 参数等于租户标识「{tenant.slug}」以确认删除",
        )

    store = build_vector_store(db)
    store.delete_tenant(tenant_id)

    # purge 会把 tenant 行也删掉，先把要写进审计的字段取出来
    tenant_name, tenant_slug = tenant.name, tenant.slug

    # 删除顺序与 flush 策略统一在 app/teardown.py 里定义，避免多处实现漂移
    purge_tenant_data(db, tenant_id)
    db.flush()

    # 说明：AuditLog 不随租户删除，保留越权/敏感操作的审计留痕（PRD 2.5）

    audit.record(
        db,
        action="platform.tenant.purge",
        tenant_id=None,
        target=tenant_id,
        ip=client_ip(request),
        detail={"slug": tenant_slug},
        commit=True,
    )
    return {"ok": True, "message": f"租户「{tenant_name}」及其全部数据已删除"}
