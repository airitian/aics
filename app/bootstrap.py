"""首次启动初始化：平台管理员 + 可选演示租户。

演示租户的作用是**让租户隔离可被肉眼验证**：两个租户各自的知识库里有
一条只属于自己的"内部信息"，互相提问必须查不到。
"""
from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import rag
from app.config import settings
from app.models import (
    AiEmployee,
    Document,
    EmployeeStatus,
    EmployeeVersion,
    KnowledgeBase,
    Message,
    Session as ChatSession,
    Tenant,
    User,
    UserRole,
)
from app.provision import create_tenant, ensure_default_employee
from app.security import hash_password
from app.utils import new_id

logger = logging.getLogger("aics.bootstrap")

DEMO_PASSWORD = "demo12345"

DEMO_TENANTS = [
    {
        "name": "星辰户外装备",
        "admin_email": "star@demo.local",
        "agent_email": "star.agent@demo.local",
        "secret": "星辰户外内部信息：老客户专属折扣码 STAR20，仅限电话下单使用。",
        "faq": [
            "配送与时效：星辰户外现货商品在付款后 48 小时内发出，使用顺丰陆运，华东地区一般 2 天送达，西北地区 3-5 天。",
            "户外电源保修：户外电源整机保修 24 个月，电池组保修 12 个月。人为损坏、进液、私自拆机不在保修范围内。",
            "发票：支持开具电子普票，下单时在备注填写抬头与税号即可；专票需提供完整开票资料，开出后 3 个工作日内发送到邮箱。",
            "退换货：收到货 7 天内不影响二次销售可无理由退货，运费由买家承担；质量问题由我们承担来回运费。",
            "帐篷防水等级：本店三人帐篷采用 3000mm 防水涂层，可应对中雨；暴雨天气建议加装天幕。",
        ],
    },
    {
        "name": "海蓝家居",
        "admin_email": "sea@demo.local",
        "agent_email": "sea.agent@demo.local",
        "secret": "海蓝家居内部信息：大客户工程单返点比例 SEA30，需线下签合同。",
        "faq": [
            "沙发送装与安装：海蓝家居沙发默认含送装一体服务，下单后 3-5 个工作日预约上门，偏远地区需额外加收运费。",
            "床垫试睡政策：床垫支持 100 天试睡，试睡期内不满意可申请退货，需保留原包装并承担退回运费。",
            "面料清洁：科技布面料可用中性清洁剂擦拭，禁止使用酒精、漂白剂，避免阳光长期直射导致褪色。",
            "尺寸定制：支持沙发尺寸定制，定制商品下单后 15 个工作日发货，定制类商品不支持无理由退货。",
            "质保：实木框架质保 5 年，五金件质保 2 年，面料与填充物质保 1 年。",
        ],
    },
]


def _publish_direct(db: Session, tenant: Tenant, employee: AiEmployee, kb_ids: list[str]) -> None:
    from app.routers.employees import SNAPSHOT_FIELDS, _serialize
    from app.utils import j_dump

    db.add(
        EmployeeVersion(
            id=new_id(),
            tenant_id=tenant.id,
            employee_id=employee.id,
            version_no=1,
            snapshot=j_dump(_serialize(employee, kb_ids, [])),
        )
    )
    employee.status = EmployeeStatus.PUBLISHED
    employee.published_version = 1


def bootstrap(db: Session) -> None:
    # 1. 平台管理员
    platform_admin = db.execute(
        select(User).where(User.role == UserRole.PLATFORM_ADMIN).limit(1)
    ).scalar_one_or_none()
    if platform_admin is None:
        db.add(
            User(
                tenant_id=None,
                email=settings.bootstrap_admin_email.strip().lower(),
                password_hash=hash_password(settings.bootstrap_admin_password),
                name="平台管理员",
                role=UserRole.PLATFORM_ADMIN,
                status="active",
            )
        )
        db.commit()
        logger.info("已创建平台管理员：%s", settings.bootstrap_admin_email)

    # 2. 演示租户
    if settings.bootstrap_demo and settings.env != "prod":
        pending: list[tuple[Document, str]] = []
        for demo in DEMO_TENANTS:
            if db.execute(select(Tenant.id).where(Tenant.name == demo["name"])).scalar_one_or_none():
                continue
            try:
                _create_demo_tenant(db, demo, pending)
            except Exception as exc:  # noqa: BLE001 - 演示数据失败不能拖垮启动
                db.rollback()
                logger.warning("演示租户「%s」创建失败（不影响服务启动）：%s", demo["name"], exc)
        # 所有演示知识库在**同一个事件循环**里入库。
        # 早先是每个租户各跑一次 asyncio.run()，结果第二次复用不到连接池不说，
        # 还会在「演示知识库入库失败：Event loop is closed」这种
        # 只有一句 warning、表现为「演示数据查不到」的地方静默失败。
        if pending:
            import asyncio

            asyncio.run(_index_demo_docs(db, pending))


def _create_demo_tenant(
    db: Session, demo: dict, pending: list[tuple[Document, str]]
) -> None:
    tenant, _ = create_tenant(
        db,
        name=demo["name"],
        admin_email=demo["admin_email"],
        admin_password=DEMO_PASSWORD,
        plan="standard",
        with_default_resources=True,
    )
    db.add(
        User(
            tenant_id=tenant.id,
            email=demo["agent_email"],
            password_hash=hash_password(DEMO_PASSWORD),
            name=f"{demo['name']} 客服",
            role=UserRole.AGENT,
            status="active",
        )
    )
    kb = db.execute(
        select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant.id).limit(1)
    ).scalar_one()
    employee = ensure_default_employee(db, tenant)
    db.commit()

    job = _prepare_demo_doc(db, tenant, kb, demo)
    if job:
        pending.append(job)

    _publish_direct(db, tenant, employee, [kb.id])
    db.commit()
    logger.info("已创建演示租户：%s", tenant.name)


def _prepare_demo_doc(
    db: Session, tenant: Tenant, kb: KnowledgeBase, demo: dict
) -> tuple[Document, str] | None:
    """只登记待入库的文档，真正入库交给 _index_demo_docs 统一调度。

    拆分是为了让所有演示文档共用一个事件循环 —— 逐个 asyncio.run 会踩到
    连接池跨循环复用的坑，失败时还只在日志里留一句 warning。
    """
    if db.execute(
        select(func.count()).select_from(Document).where(Document.tenant_id == tenant.id)
    ).scalar_one():
        return None

    doc = Document(
        id=new_id(),
        tenant_id=tenant.id,
        kb_id=kb.id,
        filename="客服常见问题.txt",
        ext=".txt",
        size_bytes=0,
        status="processing",
    )
    db.add(doc)
    db.flush()

    text = "\n\n".join(demo["faq"] + [demo["secret"]])
    doc.size_bytes = len(text.encode("utf-8"))
    kb.total_bytes = doc.size_bytes
    return doc, text


async def _index_demo_docs(db: Session, jobs: list[tuple[Document, str]]) -> None:
    """逐个入库；单个租户失败不影响其他租户（演示数据不该拖垮启动）。"""
    for doc, text in jobs:
        try:
            result = await rag.index_document(
                db, tenant_id=doc.tenant_id, kb_id=doc.kb_id, doc_id=doc.id, text=text
            )
            doc.chunk_count = result.chunk_count
            doc.status = "success"
        except Exception as exc:  # noqa: BLE001
            logger.warning("演示知识库入库失败（租户 %s）：%s", doc.tenant_id, exc)
            doc.status = "failed"
            doc.error = str(exc)
        db.commit()


def reset_demo(db: Session) -> None:
    """清空演示会话数据（方便反复验证隔离，不影响配置与知识库）。"""
    for model in (Message, ChatSession):
        for row in db.execute(select(model)).scalars().all():
            db.delete(row)
    db.commit()
