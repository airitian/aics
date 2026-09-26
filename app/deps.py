"""FastAPI 依赖：身份、租户作用域、权限、访客渠道上下文。"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.database import get_db
from app.models import Tenant, TenantStatus, User, UserRole, WidgetKey
from app.scoping import TenantScope
from app.security import Principal, decode_access_token


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _unauthorized("缺少访问令牌")
    principal = decode_access_token(authorization.split(" ", 1)[1].strip())
    if principal is None or not principal.user_id:
        raise _unauthorized("访问令牌无效或已过期")
    return principal


def get_current_user(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> User:
    user = db.get(User, principal.user_id)
    if user is None or user.status != "active":
        raise _unauthorized("账号不存在或已停用")
    return user


def require_roles(*roles: str):
    """角色或平台管理员可通过。"""

    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role == UserRole.PLATFORM_ADMIN or user.role in roles:
            return user
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="当前角色无权执行该操作")

    return _dep


def get_tenant(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> Tenant:
    """解析当前租户。

    租户标识只来自令牌中的 tid；平台运营（无租户）不得以租户身份操作，
    一律 403 并留审计，避免误用平台权限触碰租户数据。
    """
    if not principal.tenant_id:
        audit.deny(
            db,
            action="tenant.context.missing",
            actor=principal,
            target="tenant",
            detail={"reason": "令牌不含租户上下文"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前账号没有租户上下文，拒绝执行租户级操作",
        )
    tenant = db.get(Tenant, principal.tenant_id)
    if tenant is None:
        audit.deny(
            db,
            action="tenant.context.invalid",
            tenant_id=principal.tenant_id,
            actor=principal,
            target=principal.tenant_id,
            detail={"reason": "租户不存在"},
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="租户不存在或已被删除")
    if tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="租户已停用，AI 接待与配置均已冻结",
        )
    return tenant


def get_scope(
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_tenant),
) -> TenantScope:
    return TenantScope(db, tenant.id)


# --------------------------------------------------------------------------- #
# 访客侧：租户由渠道凭证「绑定关系」推导，绝不接受前端直接传租户
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WidgetContext:
    tenant: Tenant
    employee_id: str
    widget_key_id: str
    channel: str
    key: str


def get_widget_context(
    db: Session = Depends(get_db),
    x_widget_key: str | None = Header(default=None, alias="X-Widget-Key"),
) -> WidgetContext:
    if not x_widget_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少渠道凭证")
    row = db.execute(
        select(WidgetKey).where(WidgetKey.key == x_widget_key, WidgetKey.status == "active")
    ).scalar_one_or_none()
    if row is None:
        # 不区分「不存在」与「已停用」，避免探测
        audit.record(
            db,
            action="widget.key.invalid",
            result="deny",
            target=x_widget_key[:12],
            detail={"reason": "渠道凭证无效"},
            commit=True,
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="渠道凭证无效")
    tenant = db.get(Tenant, row.tenant_id)
    if tenant is None or tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该渠道暂时不可用")
    return WidgetContext(
        tenant=tenant,
        employee_id=row.employee_id,
        widget_key_id=row.id,
        channel=row.channel,
        key=row.key,
    )
