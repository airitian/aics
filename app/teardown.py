"""租户数据销毁：一处定义删除顺序，避免多处实现漂移。

为什么单独成一个模块：删除顺序是**有依赖的**（先子后父），且有两类
微妙约束，散落在路由和脚本里各写一遍必然出事：

1. **先子后父**：`messages` FK → `sessions`，`sessions` FK → `ai_employees`，
   `documents` FK → `knowledge_bases`。顺序错了直接 FK 报错。
2. **每表 flush**：部分子表在库层配了 `ON DELETE CASCADE`。若与显式 DELETE
   落在同一批 flush，同一行会被删两次，SQLAlchemy 抛
   `SAWarning: expected to delete 1 row(s); 0 were matched`。

另外两条有意为之的业务决定：
- `chunks.kb_id` 是**普通列**（不是 FK），不会被级联带走，必须显式删；
- `audit_logs` **不随租户删除**（审计留痕），且它的 tenant_id 没有 FK 约束，
  所以这里刻意不删它。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AiEmployee,
    Chunk,
    Document,
    EmployeeKbBinding,
    EmployeeVersion,
    KnowledgeBase,
    Message,
    PlaygroundRun,
    Session as ChatSession,
    Tenant,
    UsageRecord,
    User,
    VisitorMemory,
    WidgetKey,
)
from app.vectorstore import build_vector_store

# 顺序即依赖：不可随意调整
PURGE_ORDER = (
    Message,            # FK → sessions
    ChatSession,        # FK → ai_employees
    VisitorMemory,
    UsageRecord,
    PlaygroundRun,
    Chunk,              # kb_id 是普通列，不会级联删，必须显式删
    Document,           # FK → knowledge_bases
    EmployeeKbBinding,
    EmployeeVersion,
    WidgetKey,
    AiEmployee,
    KnowledgeBase,
)


def purge_tenant_data(db: Session, tenant_id: str, *, delete_tenant_row: bool = True) -> None:
    """删除单个租户的全部业务数据（含向量库中的向量）。

    调用方负责 commit。未知 tenant_id 时静默返回（幂等）。
    """
    if not tenant_id:
        raise ValueError("purge_tenant_data 必须携带 tenant_id")

    # 向量库中的向量也要清（Qdrant/pgvector 不受库级联影响）
    try:
        build_vector_store(db).delete_tenant(tenant_id)
    except Exception:  # 向量库不可达不应阻断数据销毁，但要留痕
        import logging

        logging.getLogger("aics.teardown").exception("清理向量库失败 tenant_id=%s", tenant_id)

    for model in PURGE_ORDER:
        for row in db.execute(select(model).where(model.tenant_id == tenant_id)).scalars().all():
            db.delete(row)
        db.flush()  # 逐表 flush：避免与库层 ON DELETE CASCADE 重复删同一行

    for row in db.execute(select(User).where(User.tenant_id == tenant_id)).scalars().all():
        db.delete(row)
    db.flush()

    if delete_tenant_row:
        tenant = db.get(Tenant, tenant_id)
        if tenant is not None:
            db.delete(tenant)
        db.flush()
