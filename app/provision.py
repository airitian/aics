"""租户开通与默认资源初始化。"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    AiEmployee,
    EmployeeKbBinding,
    EmployeeStatus,
    KnowledgeBase,
    Tenant,
    TenantStatus,
    User,
    UserRole,
    WidgetKey,
)
from app.security import hash_password, new_widget_key
from app.utils import new_id, slugify

DEFAULT_PERSONA = """你是「{{tenant.name}}」的在线客服 AI 员工，名字叫「{{employee.name}}」。

# 你的任务
- 依据知识库中的资料，准确、简洁地回答客户咨询
- 客户问题涉及订单、物流、售后时，先确认关键信息再给结论
- 无法确认的事情不要给结论，主动说明并询问是否需要转人工

# 表达要求
- 语气亲切专业，像真人客服，不用书面腔
- 一次回复聚焦一个问题，不超过 4 句话
- 不使用「作为AI」这类自我暴露的表述

# 限制
- 不承诺任何未在知识库中写明的优惠、时效或赔付
- 涉及投诉、纠纷、赔偿时立即转人工
"""


def create_tenant(
    db: Session,
    *,
    name: str,
    admin_email: str,
    admin_password: str,
    plan: str = "free",
    daily_token_quota: int | None = None,
    rpm_limit: int | None = None,
    timezone: str = "Asia/Shanghai",
    default_language: str = "zh-CN",
    with_default_resources: bool = True,
) -> tuple[Tenant, User]:
    email = admin_email.strip().lower()
    base_slug = slugify(name)
    slug = base_slug
    idx = 1
    while db.execute(select(Tenant.id).where(Tenant.slug == slug)).scalar_one_or_none():
        idx += 1
        slug = f"{base_slug}-{idx}"

    tenant = Tenant(
        name=name.strip(),
        slug=slug,
        status=TenantStatus.ACTIVE,
        plan=plan,
        daily_token_quota=daily_token_quota or settings.tenant_daily_tokens,
        rpm_limit=rpm_limit or settings.tenant_rpm,
        timezone=timezone,
        default_language=default_language,
    )
    db.add(tenant)
    db.flush()

    admin = User(
        tenant_id=tenant.id,
        email=email,
        password_hash=hash_password(admin_password),
        name=f"{name} 管理员",
        role=UserRole.TENANT_ADMIN,
        status="active",
    )
    db.add(admin)

    if with_default_resources:
        ensure_default_employee(db, tenant)
    db.commit()
    db.refresh(tenant)
    db.refresh(admin)
    return tenant, admin


def ensure_default_employee(db: Session, tenant: Tenant) -> AiEmployee:
    """保证每个租户至少有一个默认 AI 员工、一个默认知识库、一个渠道凭证。"""
    existing = db.execute(
        select(AiEmployee).where(
            AiEmployee.tenant_id == tenant.id, AiEmployee.is_default.is_(True)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    employee = AiEmployee(
        id=new_id(),
        tenant_id=tenant.id,
        name="默认客服",
        is_default=True,
        status=EmployeeStatus.DRAFT,
        persona=DEFAULT_PERSONA,
        time_enabled=True,
        confidence_threshold=35,
        llm_max_tokens=1200,
    )
    db.add(employee)
    db.flush()  # 先落 AI 员工，下面两张表都外键引用它

    kb = KnowledgeBase(
        id=new_id(),
        tenant_id=tenant.id,
        name="默认知识库",
        description="默认知识库，上传产品资料、FAQ 后即可被 AI 检索",
    )
    db.add(kb)
    db.flush()

    db.add(
        EmployeeKbBinding(
            id=new_id(), tenant_id=tenant.id, employee_id=employee.id, kb_id=kb.id
        )
    )
    db.add(
        WidgetKey(
            id=new_id(),
            tenant_id=tenant.id,
            employee_id=employee.id,
            key=new_widget_key(),
            name="网页插件",
            channel="web",
        )
    )
    db.flush()
    return employee


def create_employee(db: Session, tenant: Tenant, name: str, persona: str = DEFAULT_PERSONA) -> AiEmployee:
    employee = AiEmployee(
        id=new_id(),
        tenant_id=tenant.id,
        name=name.strip(),
        persona=persona,
        status=EmployeeStatus.DRAFT,
    )
    db.add(employee)
    db.flush()
    default_kb = db.execute(
        select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant.id).limit(1)
    ).scalar_one_or_none()
    if default_kb is not None:
        db.add(
            EmployeeKbBinding(
                id=new_id(), tenant_id=tenant.id, employee_id=employee.id, kb_id=default_kb.id
            )
        )
    db.flush()
    return employee
