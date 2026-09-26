"""租户作用域：所有租户级读写必须经过这里。

这是 PRD 2.3「租户上下文贯穿」的唯一收口点。设计上做两件事：
1. 把 tenant_id 做成必填位置参数 —— 想漏都漏不掉（fail closed）；
2. 查不到 / 不属于本租户时统一返回 None，不区分「不存在」与「无权访问」，避免泄露存在性。
"""
from __future__ import annotations

from typing import TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models import Base

T = TypeVar("T", bound=Base)


class TenantMismatch(Exception):
    """试图把属于别的租户的对象写入本租户作用域。"""


def scoped(stmt: Select, model: type[T], tenant_id: str) -> Select:
    """给任意 Select 加上租户过滤条件。"""
    if not tenant_id:
        raise TenantMismatch("tenant_id 为空：拒绝执行查询")
    return stmt.where(model.tenant_id == tenant_id)


def get_scoped(db: Session, model: type[T], tenant_id: str, obj_id: str) -> T | None:
    """按 id 取对象：不属于本租户一律返回 None。"""
    if not tenant_id or not obj_id:
        return None
    stmt = select(model).where(model.id == obj_id, model.tenant_id == tenant_id)
    return db.execute(stmt).scalar_one_or_none()


def list_scoped(db: Session, model: type[T], tenant_id: str, **eq_filters) -> list[T]:
    stmt = select(model).where(model.tenant_id == tenant_id)
    for key, value in eq_filters.items():
        stmt = stmt.where(getattr(model, key) == value)
    return list(db.execute(stmt).scalars().all())


def count_scoped(db: Session, model: type[T], tenant_id: str, **eq_filters) -> int:
    stmt = select(func.count()).select_from(model).where(model.tenant_id == tenant_id)
    for key, value in eq_filters.items():
        stmt = stmt.where(getattr(model, key) == value)
    return int(db.execute(stmt).scalar_one())


def add_scoped(db: Session, obj: T, tenant_id: str) -> T:
    """写入前校验归属，防止「A 租户的请求挂上 B 租户的子对象」。"""
    obj_tenant = getattr(obj, "tenant_id", None)
    if obj_tenant != tenant_id:
        raise TenantMismatch(
            f"对象租户 {obj_tenant!r} 与当前作用域 {tenant_id!r} 不一致，拒绝写入"
        )
    db.add(obj)
    return obj


def delete_scoped(db: Session, model: type[T], tenant_id: str, obj_id: str) -> bool:
    obj = get_scoped(db, model, tenant_id, obj_id)
    if obj is None:
        return False
    db.delete(obj)
    return True


class TenantScope:
    """薄封装，方便路由里 `scope.get(AiEmployee, eid)` 这样调用。"""

    __slots__ = ("db", "tenant_id")

    def __init__(self, db: Session, tenant_id: str):
        if not tenant_id:
            raise TenantMismatch("TenantScope 需要非空 tenant_id")
        self.db = db
        self.tenant_id = tenant_id

    def get(self, model: type[T], obj_id: str) -> T | None:
        return get_scoped(self.db, model, self.tenant_id, obj_id)

    def list(self, model: type[T], **eq_filters) -> list[T]:
        return list_scoped(self.db, model, self.tenant_id, **eq_filters)

    def count(self, model: type[T], **eq_filters) -> int:
        return count_scoped(self.db, model, self.tenant_id, **eq_filters)

    def add(self, obj: T) -> T:
        return add_scoped(self.db, obj, self.tenant_id)

    def delete(self, model: type[T], obj_id: str) -> bool:
        return delete_scoped(self.db, model, self.tenant_id, obj_id)
