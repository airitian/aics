"""认证路由。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.config import settings
from app.database import get_db
from app.deps import client_ip, get_current_user, get_principal
from app.models import Tenant, TenantStatus, User, UserRole
from app.schemas import LoginIn, TokenOut
from app.security import Principal, create_access_token, verify_password
from app.utils import utcnow

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/mode")
def mode() -> dict:
    """前端启动时探测：免登录模式下跳过登录页。"""
    return {"auth_disabled": settings.auth_disabled}


@router.post("/auto", response_model=TokenOut)
def auto_login(db: Session = Depends(get_db)) -> TokenOut:
    """免登录模式专用：自动签发默认租户管理员的令牌（未开启时一律 403）。"""
    if not settings.auth_disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="未开启免登录模式")
    user = (
        db.execute(
            select(User)
            .where(User.status == "active", User.role == UserRole.TENANT_ADMIN)
            .order_by(User.id)
        )
        .scalars()
        .first()
    )
    if user is None or not user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="没有可用的免登录账号")
    tenant = db.get(Tenant, user.tenant_id)
    if tenant is None or tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="免登录账号的租户不可用")
    user.last_login_at = utcnow()
    token = create_access_token(
        user_id=user.id, email=user.email, role=user.role, tenant_id=user.tenant_id
    )
    audit.record(
        db,
        action="auth.login",
        tenant_id=user.tenant_id,
        actor=Principal(user.id, user.email, user.role, user.tenant_id),
        target=user.email,
        detail={"mode": "auto"},
    )
    db.commit()
    return TokenOut(
        access_token=token,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant.name,
        name=user.name,
        email=user.email,
    )


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, request: Request, db: Session = Depends(get_db)) -> TokenOut:
    ip = client_ip(request)
    user = db.execute(select(User).where(User.email == payload.email.strip().lower())).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        audit.record(
            db,
            action="auth.login",
            result="deny",
            target=payload.email[:64],
            ip=ip,
            detail={"reason": "凭据错误"},
            commit=True,
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误")
    if user.status != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已停用")

    tenant = db.get(Tenant, user.tenant_id) if user.tenant_id else None
    if tenant is not None and tenant.status != TenantStatus.ACTIVE:
        audit.record(
            db,
            action="auth.login",
            tenant_id=tenant.id,
            result="deny",
            target=user.email,
            ip=ip,
            detail={"reason": "租户已停用"},
            commit=True,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="租户已停用，请联系服务方")

    user.last_login_at = utcnow()
    token = create_access_token(
        user_id=user.id, email=user.email, role=user.role, tenant_id=user.tenant_id
    )
    audit.record(
        db,
        action="auth.login",
        tenant_id=user.tenant_id,
        actor=Principal(user.id, user.email, user.role, user.tenant_id),
        target=user.email,
        ip=ip,
        detail={"role": user.role},
    )
    db.commit()
    return TokenOut(
        access_token=token,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant.name if tenant else None,
        name=user.name,
        email=user.email,
    )


@router.get("/me")
def me(principal: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> dict:
    user = get_current_user(principal, db)
    tenant = db.get(Tenant, user.tenant_id) if user.tenant_id else None
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "is_platform": user.role == UserRole.PLATFORM_ADMIN,
        "tenant": (
            {
                "id": tenant.id,
                "name": tenant.name,
                "slug": tenant.slug,
                "status": tenant.status,
                "plan": tenant.plan,
                "timezone": tenant.timezone,
                "default_language": tenant.default_language,
                "daily_token_quota": tenant.daily_token_quota,
                "rpm_limit": tenant.rpm_limit,
            }
            if tenant
            else None
        ),
    }
