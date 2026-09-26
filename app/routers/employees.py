"""AI 员工配置、发布与版本管理。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit, llm, prompting, ratelimit
from app.config import settings
from app.database import get_db
from app.deps import client_ip, get_scope, get_tenant, require_roles
from app.models import (
    AiEmployee,
    Document,
    EmployeeDocBinding,
    EmployeeKbBinding,
    EmployeeStatus,
    EmployeeVersion,
    KnowledgeBase,
    Tenant,
    User,
    UserRole,
    WidgetKey,
)
from app.provision import DEFAULT_PERSONA, create_employee
from app.scoping import TenantScope
from app.schemas import (
    EmployeeIn,
    EmployeePatchIn,
    PersonaDraftIn,
    PublishIn,
    PublishResultOut,
)
from app.security import new_widget_key
from app.utils import j_dump, j_load, new_id

router = APIRouter(prefix="/api", tags=["employees"])
editor = require_roles(UserRole.TENANT_ADMIN, UserRole.CONFIG_EDITOR)

SNAPSHOT_FIELDS = [
    "name", "persona", "output_language", "fast_mode", "memory_enabled",
    "auto_stop_enabled", "auto_stop_threshold", "group_reply_enabled", "group_reply_policy",
    "asr_enabled", "time_enabled", "low_confidence_policy", "confidence_threshold",
    "clarify_rounds", "llm_temperature", "llm_max_tokens", "channel_switch",
    "humanize_enabled", "reply_delay_seconds", "reply_delay_min", "reply_delay_max",
    "split_reply_enabled", "split_reply_max", "split_reply_interval_ms",
    "split_interval_min_ms", "split_interval_max_ms",
    "stop_reply_enabled", "stop_reply_rounds", "stop_condition_groups", "stop_reply_message",
    "stop_time_enabled", "stop_time_rules",
]

# 这些字段对外是对象，落库是 JSON 文本；进出都必须编解码，
# 否则 SQLite/PG 会报 "type 'dict' is not supported"。
JSON_FIELDS = {"channel_switch", "stop_time_rules", "stop_condition_groups"}


def _decode(field: str, value):
    if field in JSON_FIELDS:
        return j_load(value, {})
    return value


def _apply(employee: AiEmployee, field: str, value) -> None:
    if field in JSON_FIELDS and not isinstance(value, str):
        value = j_dump(value)
    setattr(employee, field, value)


def _serialize(employee: AiEmployee, kb_ids: list[str], doc_ids: list[str]) -> dict:
    data = {f: _decode(f, getattr(employee, f)) for f in SNAPSHOT_FIELDS}
    # 旧数据回退：区间列还是初始默认值时，取旧单值字段，保证前端拿到有效区间
    if not data["reply_delay_min"] and not data["reply_delay_max"] and data["reply_delay_seconds"]:
        data["reply_delay_min"] = data["reply_delay_max"] = data["reply_delay_seconds"]
    if (
        data["split_interval_min_ms"] == 800 and data["split_interval_max_ms"] == 800
        and data["split_reply_interval_ms"] != 800
    ):
        data["split_interval_min_ms"] = data["split_interval_max_ms"] = data["split_reply_interval_ms"]
    data["kb_ids"] = kb_ids
    data["doc_ids"] = doc_ids
    return data


def _kb_ids(db: Session, tenant_id: str, employee_id: str) -> list[str]:
    return list(
        db.execute(
            select(EmployeeKbBinding.kb_id).where(
                EmployeeKbBinding.tenant_id == tenant_id,
                EmployeeKbBinding.employee_id == employee_id,
            )
        ).scalars().all()
    )


def _doc_ids(db: Session, tenant_id: str, employee_id: str) -> list[str]:
    return list(
        db.execute(
            select(EmployeeDocBinding.doc_id).where(
                EmployeeDocBinding.tenant_id == tenant_id,
                EmployeeDocBinding.employee_id == employee_id,
            )
        ).scalars().all()
    )


def _bind_kbs(db: Session, tenant_id: str, employee_id: str, kb_ids: list[str]) -> None:
    valid = set(
        db.execute(
            select(KnowledgeBase.id).where(
                KnowledgeBase.tenant_id == tenant_id, KnowledgeBase.id.in_(kb_ids or [])
            )
        ).scalars().all()
    )
    invalid = [k for k in (kb_ids or []) if k not in valid]
    if invalid:
        # 绑定了别的租户或不存在的知识库 —— 直接拒绝，避免把别人的库挂上来
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"以下知识库不存在或不属于本租户：{invalid}",
        )
    for row in db.execute(
        select(EmployeeKbBinding).where(
            EmployeeKbBinding.tenant_id == tenant_id, EmployeeKbBinding.employee_id == employee_id
        )
    ).scalars().all():
        db.delete(row)
    # 必须先落删除再插新绑定：默认 flush 顺序是 INSERT 在 DELETE 之前，
    # 否则会撞 (employee_id, kb_id) 唯一约束。
    db.flush()
    for kb_id in valid:
        db.add(
            EmployeeKbBinding(
                id=new_id(), tenant_id=tenant_id, employee_id=employee_id, kb_id=kb_id
            )
        )


def _bind_docs(db: Session, tenant_id: str, employee_id: str, doc_ids: list[str]) -> None:
    """绑定文件级检索范围。校验文档属于本租户，防止挂载别人的文件。"""
    valid = set(
        db.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id, Document.id.in_(doc_ids or [])
            )
        ).scalars().all()
    )
    invalid = [d for d in (doc_ids or []) if d not in valid]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"以下文档不存在或不属于本租户：{invalid}",
        )
    for row in db.execute(
        select(EmployeeDocBinding).where(
            EmployeeDocBinding.tenant_id == tenant_id, EmployeeDocBinding.employee_id == employee_id
        )
    ).scalars().all():
        db.delete(row)
    # 同 _bind_kbs：先删后插，避开唯一约束的 flush 顺序问题
    db.flush()
    for doc_id in valid:
        db.add(
            EmployeeDocBinding(
                id=new_id(), tenant_id=tenant_id, employee_id=employee_id, doc_id=doc_id
            )
        )


def _out(db: Session, employee: AiEmployee) -> dict:
    kb_ids = _kb_ids(db, employee.tenant_id, employee.id)
    doc_ids = _doc_ids(db, employee.tenant_id, employee.id)
    return {
        **_serialize(employee, kb_ids, doc_ids),
        "id": employee.id,
        "status": employee.status,
        "is_default": employee.is_default,
        "published_version": employee.published_version,
        "persona_chars": len(employee.persona or ""),
        "persona_max_chars": settings.persona_max_chars,
        "updated_at": employee.updated_at.isoformat(),
    }


@router.get("/employees")
def list_employees(
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    rows = scope.list(AiEmployee)
    rows.sort(key=lambda e: (not e.is_default, e.created_at))
    return {"items": [_out(scope.db, e) for e in rows], "max": settings.max_employees_per_tenant}


@router.post("/employees")
def create(
    payload: EmployeeIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    current = scope.count(AiEmployee)
    if current >= settings.max_employees_per_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"单租户 AI 员工数已达上限（{settings.max_employees_per_tenant}）",
        )
    employee = create_employee(scope.db, tenant, payload.name, payload.persona or DEFAULT_PERSONA)
    for key, value in payload.model_dump(exclude={"name", "persona", "kb_ids", "doc_ids", "is_default"}).items():
        _apply(employee, key, value)
    if payload.is_default:
        _clear_default(scope, employee.id)
        employee.is_default = True
    _bind_kbs(scope.db, tenant.id, employee.id, payload.kb_ids)
    _bind_docs(scope.db, tenant.id, employee.id, payload.doc_ids)
    scope.db.commit()
    audit.record(
        scope.db, action="employee.create", tenant_id=tenant.id,
        actor=None, target=employee.id, ip=client_ip(request), detail={"name": employee.name}, commit=True,
    )
    return _out(scope.db, employee)


# --------------------------------------------------------------------------- #
# 人设智能生成 / 优化
# --------------------------------------------------------------------------- #
_PERSONA_SYSTEM_PROMPT = """你是资深的客服运营专家，为在线客服系统的 AI 员工撰写「人设」（即 system prompt）。

输出要求：
1. 只输出人设正文本身，不要任何解释、前言或 Markdown 代码块包裹。
2. 用以下结构组织：# 角色（一段话定位）、## 任务目标（3-6 条编号）、## 语言风格（2-4 条）、## 行为约束（必须包含：只依据知识库内容回答，知识库没有的信息如实告知并引导人工，绝不编造）。
3. 语言通顺、具体、可执行，避免空话套话；总长度控制在 800 字以内。"""


def _persona_style_line(style: str, length: str) -> str:
    hints = []
    style_map = {
        "友好": "语气友好热情，让客户感到被欢迎",
        "自然": "表达自然口语化，像真人聊天",
        "柔和": "语气柔和耐心，善用缓冲与共情表达",
        "专业": "表达严谨专业，用词准确",
        "幽默": "在合适场合带一点轻松幽默，但不失分寸",
    }
    length_map = {
        "简洁": "回复尽量简短，一两句话说到重点",
        "标准": "回复长度适中，重点完整、不冗长",
        "详细": "回复可以更充分，把步骤和注意事项讲清楚",
    }
    if style in style_map:
        hints.append(style_map[style])
    if length in length_map:
        hints.append(length_map[length])
    if not hints:
        return ""
    return "\n语言风格附加要求：" + "；".join(hints) + "。"


@router.post("/employees/persona-draft")
async def persona_draft(
    payload: PersonaDraftIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    if payload.mode == "generate" and not payload.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="请先填写「您希望 AI 员工做什么」"
        )
    if payload.mode == "optimize" and not payload.current_persona.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="当前人设为空，请改用「生成人设」"
        )

    lang = payload.output_language or "zh-CN"
    style_line = _persona_style_line(payload.style, payload.length)
    if payload.mode == "generate":
        user_prompt = (
            f"请为满足以下需求的客服 AI 员工撰写人设（用{lang}撰写）：\n"
            f"【需求描述】{payload.description.strip()}\n"
            f"{style_line}"
        ).strip()
    else:
        user_prompt = (
            f"请在保留原有人设意图的基础上优化以下人设（用{lang}输出）：\n\n"
            f"【当前人设】\n{payload.current_persona.strip()}\n\n"
            f"【优化要求】{payload.optimize_note.strip() or '整体优化：结构更清晰、表述更具体、约束更完善'}"
            f"{style_line}"
        ).strip()

    try:
        result = await llm.chat(
            [
                {"role": "system", "content": _PERSONA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=2000,
        )
    except llm.LLMUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    text = (result.text or "").strip()
    # 剥掉模型偶尔自作主张包上的 ``` 代码块围栏
    if text.startswith("```"):
        text = text.strip("`").lstrip()
        if text.lower().startswith("markdown"):
            text = text[8:]
        text = text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_502_BAD_REQUEST, detail="模型返回了空内容，请重试")

    ratelimit.record_usage(
        scope.db,
        tenant_id=tenant.id,
        employee_id=None,
        kind="llm",
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        origin="persona",
        commit=True,
    )
    return {
        "persona": text[: settings.persona_max_chars],
        "model": result.model,
        "endpoint": result.endpoint,
        "degraded": result.degraded,
        "tokens": result.total_tokens,
    }


def _clear_default(scope: TenantScope, keep_id: str | None = None) -> None:
    for row in scope.list(AiEmployee):
        if keep_id is not None and row.id == keep_id:
            continue
        row.is_default = False


@router.get("/employees/{employee_id}")
def detail(
    employee_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    return _out(scope.db, employee)


@router.patch("/employees/{employee_id}")
def update(
    employee_id: str,
    payload: EmployeePatchIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    changes = payload.model_dump(exclude_unset=True)

    if changes.pop("is_default", False):
        _clear_default(scope, employee.id)
        employee.is_default = True

    kb_ids = changes.pop("kb_ids", None)
    doc_ids = changes.pop("doc_ids", None)
    config_changed = bool(changes)
    for key, value in changes.items():
        _apply(employee, key, value)
    # 区间字段防呆：部分更新后可能出现 min>max，交换归一，避免脏状态
    if employee.reply_delay_min > employee.reply_delay_max:
        employee.reply_delay_min, employee.reply_delay_max = (
            employee.reply_delay_max, employee.reply_delay_min,
        )
    if employee.split_interval_min_ms > employee.split_interval_max_ms:
        employee.split_interval_min_ms, employee.split_interval_max_ms = (
            employee.split_interval_max_ms, employee.split_interval_min_ms,
        )
    if kb_ids is not None:
        _bind_kbs(scope.db, tenant.id, employee.id, kb_ids)
        config_changed = True
    if doc_ids is not None:
        _bind_docs(scope.db, tenant.id, employee.id, doc_ids)
        config_changed = True
    if config_changed:
        # 改完配置即回到草稿态，必须再次发布才对线上生效（PRD 3.2）
        employee.status = EmployeeStatus.DRAFT

    scope.db.commit()
    return _out(scope.db, employee)


@router.delete("/employees/{employee_id}")
def remove(
    employee_id: str,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    if employee.is_default and scope.count(AiEmployee) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="默认 AI 员工不可删除，请先把其他员工设为默认",
        )
    scope.db.delete(employee)
    scope.db.commit()
    audit.record(
        scope.db, action="employee.delete", tenant_id=tenant.id, target=employee_id,
        ip=client_ip(request), detail={"name": employee.name}, commit=True,
    )
    return {"ok": True}


# --------------------------------------------------------------------------- #
# 发布与版本
# --------------------------------------------------------------------------- #
def _validate_for_publish(
    db: Session, employee: AiEmployee, kb_ids: list[str], doc_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """返回 (阻断项, 需二次确认项, 提示项)。"""
    blocked: list[str] = []
    confirm: list[str] = []
    notes: list[str] = []

    if not (employee.persona or "").strip():
        blocked.append("人设不能为空")

    unknown = prompting.unknown_variables(employee.persona or "")
    if unknown:
        blocked.append(f"人设引用了不存在的变量：{'、'.join(unknown)}")

    if settings.llm_provider == "api" and not (
        settings.llm_base_url and settings.llm_api_key and settings.llm_model
    ):
        blocked.append("未配置可用的对话模型端点")
    elif settings.llm_provider == "stub":
        notes.append("当前为开发模式（LLM_PROVIDER=stub），未接入真实对话模型")

    if settings.embed_provider == "local":
        notes.append("当前使用本地占位向量模型（EMBED_PROVIDER=local），不具备真实语义能力")

    doc_count = 0
    if doc_ids:
        doc_count = int(
            db.execute(
                select(func.count())
                .select_from(Document)
                .where(
                    Document.tenant_id == employee.tenant_id,
                    Document.id.in_(doc_ids),
                    Document.enabled.is_(True),
                )
            ).scalar_one()
        )
    elif kb_ids:
        doc_count = int(
            db.execute(
                select(func.count())
                .select_from(Document)
                .where(
                    Document.tenant_id == employee.tenant_id,
                    Document.kb_id.in_(kb_ids),
                    Document.enabled.is_(True),
                )
            ).scalar_one()
        )
    if not doc_ids and not kb_ids:
        confirm.append("未绑定任何知识库或文件，AI 将无法回答业务问题")
    elif doc_count == 0:
        confirm.append("绑定的知识库或文件中没有任何已启用内容，AI 将无法回答业务问题")
    return blocked, confirm, notes


@router.post("/employees/{employee_id}/publish", response_model=PublishResultOut)
def publish(
    employee_id: str,
    payload: PublishIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> PublishResultOut:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")

    kb_ids = _kb_ids(scope.db, tenant.id, employee.id)
    doc_ids = _doc_ids(scope.db, tenant.id, employee.id)
    blocked, confirm, notes = _validate_for_publish(scope.db, employee, kb_ids, doc_ids)
    if blocked:
        audit.record(
            scope.db, action="employee.publish", tenant_id=tenant.id, target=employee_id,
            result="deny", ip=client_ip(request), detail={"blocked": blocked}, commit=True,
        )
        return PublishResultOut(ok=False, blocked=blocked, warnings=notes, message="发布被阻断")

    if confirm and not payload.force:
        # 空知识库等场景：允许发布，但必须二次确认（PRD 3.2）
        return PublishResultOut(
            ok=False,
            blocked=[],
            warnings=confirm + notes,
            message="存在需要确认的项，确认无误后请以 force=true 再次提交",
        )

    last_no = int(
        scope.db.execute(
            select(func.coalesce(func.max(EmployeeVersion.version_no), 0)).where(
                EmployeeVersion.tenant_id == tenant.id,
                EmployeeVersion.employee_id == employee.id,
            )
        ).scalar_one()
    )
    version_no = last_no + 1
    scope.db.add(
        EmployeeVersion(
            id=new_id(),
            tenant_id=tenant.id,
            employee_id=employee.id,
            version_no=version_no,
            snapshot=_snapshot_json(employee, kb_ids, doc_ids),
            created_by=user.id,
        )
    )
    employee.status = EmployeeStatus.PUBLISHED
    employee.published_version = version_no
    scope.db.flush()

    # 只保留最近 N 个版本
    old = scope.db.execute(
        select(EmployeeVersion)
        .where(
            EmployeeVersion.tenant_id == tenant.id, EmployeeVersion.employee_id == employee.id
        )
        .order_by(EmployeeVersion.version_no.desc())
        .offset(settings.max_versions_kept)
    ).scalars().all()
    for row in old:
        scope.db.delete(row)

    audit.record(
        scope.db, action="employee.publish", tenant_id=tenant.id, target=employee_id,
        ip=client_ip(request), detail={"version": version_no}, commit=True,
    )
    return PublishResultOut(
        ok=True, version_no=version_no, warnings=notes, message=f"已发布为版本 v{version_no}"
    )


def _snapshot_json(employee: AiEmployee, kb_ids: list[str], doc_ids: list[str]) -> str:
    from app.utils import j_dump

    return j_dump(_serialize(employee, kb_ids, doc_ids))


@router.get("/employees/{employee_id}/versions")
def versions(
    employee_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    rows = scope.db.execute(
        select(EmployeeVersion)
        .where(
            EmployeeVersion.tenant_id == employee.tenant_id,
            EmployeeVersion.employee_id == employee_id,
        )
        .order_by(EmployeeVersion.version_no.desc())
    ).scalars().all()
    return {
        "items": [
            {
                "version_no": r.version_no,
                "created_at": r.created_at.isoformat(),
                "is_current": r.version_no == employee.published_version,
                "snapshot": r.data,
            }
            for r in rows
        ]
    }


@router.post("/employees/{employee_id}/versions/{version_no}/rollback")
def rollback(
    employee_id: str,
    version_no: int,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    version = scope.db.execute(
        select(EmployeeVersion).where(
            EmployeeVersion.tenant_id == tenant.id,
            EmployeeVersion.employee_id == employee_id,
            EmployeeVersion.version_no == version_no,
        )
    ).scalar_one_or_none()
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="版本不存在")

    data = version.data
    for field in SNAPSHOT_FIELDS:
        if field in data:
            _apply(employee, field, data[field])
    _bind_kbs(scope.db, tenant.id, employee.id, data.get("kb_ids") or [])
    _bind_docs(scope.db, tenant.id, employee.id, data.get("doc_ids") or [])
    employee.status = EmployeeStatus.DRAFT
    scope.db.commit()

    audit.record(
        scope.db, action="employee.rollback", tenant_id=tenant.id, target=employee_id,
        ip=client_ip(request), detail={"to_version": version_no, "published_now": employee.published_version},
        commit=True,
    )
    return {
        "ok": True,
        "message": f"已把草稿回滚为 v{version_no} 的内容，**需再次发布**才会对线上生效",
        "published_version": employee.published_version,
    }


# --------------------------------------------------------------------------- #
# 渠道凭证
# --------------------------------------------------------------------------- #
@router.get("/employees/{employee_id}/widget-keys")
def list_widget_keys(
    employee_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    rows = scope.list(WidgetKey, employee_id=employee_id)
    return {
        "items": [
            {
                "id": r.id,
                "key": r.key,
                "name": r.name,
                "channel": r.channel,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    }


@router.post("/employees/{employee_id}/widget-keys")
def create_widget_key(
    employee_id: str,
    name: str = "网页插件",
    channel: str = "web",
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    employee = scope.get(AiEmployee, employee_id)
    if employee is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
    row = WidgetKey(
        id=new_id(),
        tenant_id=employee.tenant_id,
        employee_id=employee_id,
        key=new_widget_key(),
        name=name[:120],
        channel=channel[:32],
    )
    scope.add(row)
    scope.db.commit()
    return {"id": row.id, "key": row.key, "name": row.name, "channel": row.channel, "status": row.status}


@router.delete("/widget-keys/{key_id}")
def delete_widget_key(
    key_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    if not scope.delete(WidgetKey, key_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="渠道凭证不存在")
    scope.db.commit()
    return {"ok": True}
