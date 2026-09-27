"""访客侧对话接口。

租户来源：X-Widget-Key → widget_keys 表 → (tenant_id, employee_id)。
**绝不接受请求体里的 tenant_id**（PRD 2.3）。

延迟回复（聚合窗口）：开启后访客消息先进内存缓冲，等 `reply_delay_seconds`
秒内没有**更新的**消息时，由最后一条消息的请求把窗口内所有消息合并为一条
访客消息统一生成回复；窗口内更早的请求静默返回（reply 为空，前端不渲染）。
窗口内消息只存在内存里，服务重启会丢（窗口只有几十秒，可接受）；
进程内单事件循环串行检查，无并发竞争。
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import agent as agent_mod
from app import chatimages, prompting, ratelimit
from app.config import settings
from app.database import get_db
from app.deps import WidgetContext, get_widget_context
from app.docstore import read_chat_image, save_chat_image
from app.models import AiEmployee, ChatImage, Message, MessageRole, Session as ChatSession, SessionStatus
from app.schemas import ChatIn, ChatOut, HitOut
from app.utils import new_id, utcnow

logger = logging.getLogger("aics.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

OPEN_STATUSES = (SessionStatus.AI, SessionStatus.QUEUED, SessionStatus.HUMAN)

# 延迟回复的聚合窗口：session_id -> [(到达时刻, 消息文本, 图片载荷)]，按到达顺序
_PENDING: dict[str, list[tuple[float, str, list[dict]]]] = {}


def _reply_delay_seconds(employee: AiEmployee) -> int:
    """本次回复的延迟秒数：在 [min, max] 区间随机，更像真人。

    旧数据回退：区间列还是 0/0 时取旧字段 reply_delay_seconds；0 表示关闭。
    """
    lo, hi = employee.reply_delay_min, employee.reply_delay_max
    if not lo and not hi:
        lo = hi = employee.reply_delay_seconds or 0
    if hi <= 0:
        return 0
    return random.randint(min(lo, hi), hi)


def _split_interval_ms(employee: AiEmployee) -> int:
    """拆分消息的发送间隔：在 [min, max] 区间随机；未开启拆分时固定 800。"""
    if not employee.split_reply_enabled:
        return 800
    lo, hi = employee.split_interval_min_ms, employee.split_interval_max_ms
    if lo == 800 and hi == 800 and employee.split_reply_interval_ms != 800:
        lo = hi = employee.split_reply_interval_ms  # 旧数据回退
    return random.randint(min(lo, hi), max(lo, hi))


def _employee(db: Session, ctx: WidgetContext) -> AiEmployee:
    employee = db.get(AiEmployee, ctx.employee_id)
    # 二次校验：凭证绑定的员工必须属于凭证所属租户
    if employee is None or employee.tenant_id != ctx.tenant.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="渠道凭证与 AI 员工不匹配"
        )
    return employee


@router.get("/widget-config")
def widget_config(
    ctx: WidgetContext = Depends(get_widget_context),
    db: Session = Depends(get_db),
) -> dict:
    employee = _employee(db, ctx)
    return {
        "tenant_name": ctx.tenant.name,
        "employee_name": employee.name,
        "channel": ctx.channel,
        "greeting": f"您好，我是{ctx.tenant.name}的在线客服{employee.name}，请问有什么可以帮您？",
        "asr_enabled": employee.asr_enabled,
        "split_interval_ms": _split_interval_ms(employee) if employee.split_reply_enabled else 800,
        "published": employee.published_version > 0,
    }


@router.post("/images")
async def upload_image(
    visitor_id: str = Form(...),
    file: UploadFile = File(...),
    ctx: WidgetContext = Depends(get_widget_context),
    db: Session = Depends(get_db),
) -> dict:
    """上传一张聊天图片，返回 image_id；发送消息时放进 image_ids。

    上传与发送分离：图片要先有 id，消息体（JSON）才能引用它。
    此刻只做落盘，转写推迟到发送时 —— 大多数人传了图还会打字，
    在上传时转写会白白多等一次视觉模型（4s 级）。
    """
    data = await file.read()
    ext = chatimages.validate_upload(file, data)

    img = ChatImage(
        id=new_id(),
        tenant_id=ctx.tenant.id,
        visitor_id=visitor_id[:64],
        filename=(file.filename or "")[:255],
        mime=(file.content_type or "").split(";")[0].strip().lower(),
        ext=ext,
        size_bytes=len(data),
    )
    db.add(img)
    db.flush()
    img.rel_path = await run_in_threadpool(save_chat_image, ctx.tenant.id, img.id, ext, data)
    db.commit()
    return {"image_id": img.id, "url": chatimages.url_of(img.id, visitor_id=visitor_id)}


@router.get("/images/{image_id}")
def get_image(
    image_id: str,
    visitor_id: str,
    exp: int,
    sig: str,
    db: Session = Depends(get_db),
) -> Response:
    """取回聊天图片原图。访客只能取自己的图（同会话回放需要）。

    `<img src>` 无法携带 X-Widget-Key 头，因此这里不做请求头认证，
    改为校验 chatimages.sign_url 签发的 HMAC 签名 + 过期时间。
    """
    if not chatimages.verify_sig(image_id, f"v:{visitor_id}", exp, sig):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="链接无效或已过期")
    row = db.execute(
        select(ChatImage).where(
            ChatImage.id == image_id,
            ChatImage.visitor_id == visitor_id,
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片不存在")
    data = read_chat_image(row.rel_path)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片文件已丢失")
    return Response(content=data, media_type=row.mime or "image/png")


def _get_or_create_session(
    db: Session, *, ctx: WidgetContext, payload: ChatIn
) -> ChatSession:
    tenant_id = ctx.tenant.id
    if payload.session_id:
        row = db.execute(
            select(ChatSession).where(
                ChatSession.id == payload.session_id,
                ChatSession.tenant_id == tenant_id,          # 必须带租户条件
                ChatSession.visitor_id == payload.visitor_id,  # 且必须属于同一访客
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        return row

    row = db.execute(
        select(ChatSession)
        .where(
            ChatSession.tenant_id == tenant_id,
            ChatSession.visitor_id == payload.visitor_id,
            ChatSession.employee_id == ctx.employee_id,
            ChatSession.status.in_(OPEN_STATUSES),
        )
        .order_by(ChatSession.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is not None:
        return row

    row = ChatSession(
        tenant_id=tenant_id,
        employee_id=ctx.employee_id,
        visitor_id=payload.visitor_id,
        channel=ctx.channel or payload.channel,
        status=SessionStatus.AI,
    )
    db.add(row)
    db.flush()
    return row


@router.post("/message", response_model=ChatOut)
async def send_message(
    payload: ChatIn,
    ctx: WidgetContext = Depends(get_widget_context),
    db: Session = Depends(get_db),
) -> ChatOut:
    tenant = ctx.tenant
    employee = _employee(db, ctx)

    try:
        ratelimit.limiter.acquire(tenant.id, tenant.rpm_limit, settings.tenant_concurrent)
    except ratelimit.RateLimited as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    try:
        session = _get_or_create_session(db, ctx=ctx, payload=payload)

        # 图片：归属校验 + 首次绑定会话 + 转写（失败降级为「读不到图」，不拦消息）
        image_payloads = await chatimages.resolve_and_transcribe(
            db,
            tenant_id=tenant.id,
            visitor_id=payload.visitor_id,
            session_id=session.id,
            image_ids=payload.image_ids,
        )
        image_meta = chatimages.meta_images(image_payloads)

        # 人工接管中：AI 静默，访客消息照常落库（不丢消息），不给 AI 回复（PRD 6.1）
        if session.status == SessionStatus.HUMAN:
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.VISITOR,
                    content=payload.message,
                    meta=f'{{"images": {json.dumps(image_meta, ensure_ascii=False)}}}',
                )
            )
            session.last_active_at = utcnow()
            db.commit()
            return ChatOut(
                session_id=session.id,
                reply="",
                status=session.status,
                handoff=True,
                handoff_reason="人工客服正在接待",
                degraded=True,
                degrade_reason="human_takeover",
            )

        # ---- 延迟回复（聚合窗口）----
        message_text = payload.message
        delay = _reply_delay_seconds(employee)
        if delay > 0 and session.status == SessionStatus.AI:
            arrival = time.monotonic()
            _PENDING.setdefault(session.id, []).append((arrival, payload.message, image_meta))
            await asyncio.sleep(delay)
            buf = _PENDING.get(session.id) or []
            if not buf or buf[-1][0] != arrival:
                # 窗口内来了更新的消息：由最后一条消息的请求统一生成，本请求静默返回
                return ChatOut(session_id=session.id, reply="", status=session.status)
            # 我是窗口内最后一条：取走全部缓冲消息，合并为一条访客消息
            _PENDING[session.id] = []
            message_text = "\n".join(t for _, t, _ in buf) or payload.message
            # 窗口内可能多条都带了图：合并引用（转写文本仍在 ChatImage，不重复存储）
            merged_images = [im for _, _, ims in buf for im in ims][: settings.chat_image_max_per_message]
            image_meta = merged_images
            image_payloads = await chatimages.resolve_and_transcribe(
                db,
                tenant_id=tenant.id,
                visitor_id=payload.visitor_id,
                session_id=session.id,
                image_ids=[im["id"] for im in merged_images],
            )
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.VISITOR,
                    content=message_text,
                    meta=json.dumps(
                        {"merged": len(buf), "images": image_meta}, ensure_ascii=False
                    ),
                )
            )
        else:
            # 访客消息先落库，保证不丢消息
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.VISITOR,
                    content=payload.message,
                    meta=json.dumps({"images": image_meta}, ensure_ascii=False) if image_meta else "{}",
                )
            )
        session.last_active_at = utcnow()

        # 配额耗尽：本租户降级转人工，其他租户不受影响（PRD 2.5）
        try:
            ratelimit.ensure_token_quota(db, tenant, estimated=employee.llm_max_tokens)
        except ratelimit.QuotaExceeded as exc:
            session.status = SessionStatus.QUEUED
            session.handoff_reason = "本租户模型额度已用尽"
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.SYSTEM,
                    content=exc.message,
                )
            )
            reply = "已为您转接人工客服，请稍候。"
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.AI,
                    content=reply,
                )
            )
            db.commit()
            return ChatOut(
                session_id=session.id,
                reply=reply,
                status=session.status,
                handoff=True,
                handoff_reason=session.handoff_reason,
                degraded=True,
                degrade_reason="quota_exceeded",
            )

        result = await agent_mod.handle_turn(
            db, tenant=tenant, employee=employee, session=session, message=message_text,
            images=image_payloads or None,
        )

        # 连续未有效回答 → 达阈值则停止 AI 回复并转人工（PRD 3.1.5）
        if result.degraded and not result.handoff and "低置信" in (result.degrade_reason or ""):
            if agent_mod.escalate_no_answer(session, employee):
                result.handoff = True
                result.handoff_reason = session.handoff_reason
                result.reply = (
                    prompting.HANDOFF_REPLY
                    if agent_mod.has_online_agent(db, tenant.id)
                    else prompting.HANDOFF_QUEUED_REPLY
                )

        # 补齐「AI 实际说的话」：早退分支（无意义输入 / 辱骂 / 广告 / 低置信追问 /
        # 转人工 / 模型不可用）只写了 SYSTEM 说明，甚至什么都不写 —— 于是访客
        # 收到了回复，消息表里却没有，历史回放与实际对话不一致。
        # 放在这里统一补，保证「访客看到的」与「记录下来的」永远一致。
        if result.reply and not result.ai_saved:
            db.add(
                Message(
                    tenant_id=tenant.id,
                    session_id=session.id,
                    role=MessageRole.AI,
                    content=result.reply,
                )
            )

        db.commit()
        return ChatOut(
            session_id=session.id,
            reply=result.reply,
            status=session.status,
            handoff=result.handoff,
            handoff_reason=result.handoff_reason,
            hits=[
                HitOut(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    kb_id=h.kb_id,
                    filename=h.filename,
                    score=round(h.score, 4),
                    text=h.text,
                )
                for h in result.hits
            ],
            degraded=result.degraded,
            degrade_reason=result.degrade_reason,
            usage_total_tokens=result.tokens,
            segments=result.segments,
        )
    finally:
        ratelimit.limiter.release(tenant.id)


@router.get("/history")
def history(
    visitor_id: str,
    session_id: str | None = None,
    limit: int = 50,
    ctx: WidgetContext = Depends(get_widget_context),
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(ChatSession).where(
        ChatSession.tenant_id == ctx.tenant.id,
        ChatSession.visitor_id == visitor_id,
    )
    if session_id:
        stmt = stmt.where(ChatSession.id == session_id)
    session = db.execute(stmt.order_by(ChatSession.created_at.desc()).limit(1)).scalar_one_or_none()
    if session is None:
        return {"session_id": None, "items": []}

    rows = (
        db.execute(
            select(Message)
            .where(Message.tenant_id == ctx.tenant.id, Message.session_id == session.id)
            .order_by(Message.created_at.desc())
            .limit(min(200, max(1, limit)))
        )
        .scalars()
        .all()
    )
    return {
        "session_id": session.id,
        "status": session.status,
        "items": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "images": [
                    {"id": i["id"], "url": chatimages.url_of(i["id"], visitor_id=visitor_id)}
                    for i in (m.meta_data.get("images") or [])
                    if i.get("id")
                ],
            }
            for m in reversed(rows)
            if m.content or (m.meta_data.get("images") or [])
        ],
    }
