"""客户聊天图片：上传校验 → 落盘 → 视觉转写 → 归属校验。

为什么独立成模块
----------------
chat（访客挂件）与 testchat（管理后台对话测试）走的是**两套认证、两个入口**，
但图片的处理逻辑必须完全一致 —— 不然「测试页测出来的行为」和「线上行为」
就会分叉，测试就骗人了。两边都只调这里，转写/校验只写一遍。

失败边界
--------
- 上传：大小/类型超限直接 4xx（此刻还没消耗任何视觉配额）；
- 转写：视觉模型失败**不让消息发送失败** —— 客户发的图读不出来，
  正确行为是 AI 如实说「图里的内容我看不到」，而不是整条消息 500。
  转写结果缓存在 ChatImage.ocr_text，多轮对话不重复消耗视觉配额。
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import docstore, vision
from app.config import settings
from app.models import ChatImage, ChatImageStatus

logger = logging.getLogger("aics.chatimages")

# 与 vision/文档资产一致的图片类型白名单（HEIC 不收：多数视觉接口不认，
# 让前端/客户端先转 JPEG）
ALLOWED_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
}


def validate_upload(file: UploadFile, data: bytes) -> str:
    """上传校验，返回扩展名。不合规抛 HTTPException。"""
    mime = (file.content_type or "").split(";")[0].strip().lower()
    ext = ALLOWED_MIMES.get(mime)
    if ext is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="仅支持 png / jpg / webp / gif / bmp 图片",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空")
    if len(data) > settings.chat_image_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"图片超过 {settings.chat_image_max_bytes // 1024 // 1024}MB，请压缩后重试",
        )
    return ext


async def resolve_and_transcribe(
    db: Session,
    *,
    tenant_id: str,
    visitor_id: str,
    session_id: str,
    image_ids: list[str],
) -> list[dict]:
    """把一条消息携带的 image_ids 处理成 handle_turn 需要的载荷。

    - 归属校验：图必须属于本租户 + 本访客（拿别人的 image_id 注入上下文 = 越权）；
    - 首次发送时绑定 session_id（上传时可能还没有会话）；
    - 转写：只对 pending 的图调视觉模型，结果缓存 —— 多轮引用同一张图不再重识别。
    无效的 image_id 静默跳过：图片缺失降级为「少一张图的上下文」，
    不能让整条消息发不出去。
    """
    ids = list(dict.fromkeys(i for i in image_ids if i))[: settings.chat_image_max_per_message]
    if not ids:
        return []

    payloads: list[dict] = []
    dirty = False
    for img_id in ids:
        row = db.execute(
            select(ChatImage).where(
                ChatImage.id == img_id,
                ChatImage.tenant_id == tenant_id,
                ChatImage.visitor_id == visitor_id,
            )
        ).scalar_one_or_none()
        if row is None:
            logger.warning("聊天图片归属校验失败 tenant=%s visitor=%s image=%s", tenant_id, visitor_id, img_id)
            continue
        if row.session_id != session_id:
            row.session_id = session_id
            dirty = True
        if row.ocr_status == ChatImageStatus.PENDING:
            data = await run_in_threadpool(docstore.read_chat_image, row.rel_path)
            if data:
                text = await vision.describe(data, row.mime, mode="chat")
                if text:
                    row.ocr_text, row.ocr_status = text, ChatImageStatus.OK
                else:
                    row.ocr_status = ChatImageStatus.FAILED
            else:
                row.ocr_status = ChatImageStatus.FAILED
            dirty = True
        payloads.append({"id": row.id, "ocr_text": row.ocr_text, "ocr_status": row.ocr_status})
    if dirty:
        db.flush()
    return payloads


def meta_images(payloads: list[dict]) -> list[dict]:
    """写进 Message.meta 的轻量引用（转写全文存 ChatImage，不复制进每条消息）。"""
    return [{"id": p["id"], "ocr_status": p["ocr_status"]} for p in payloads]


def url_of(image_id: str, *, visitor_id: str = "") -> str:
    """挂件侧图片 URL；后台侧用 /api/testchat 的图端点，由后台自己拼。"""
    from urllib.parse import quote

    return f"/api/chat/images/{image_id}?visitor_id={quote(visitor_id)}"
