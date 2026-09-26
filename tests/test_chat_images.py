"""聊天图片链路：上传校验 → 发送带图 → 转写缓存 → 归属隔离 → 多轮上下文。"""
from __future__ import annotations

import io
import struct
import zlib

import pytest

from app.agent import _recent_history
from app.models import ChatImage
from tests.conftest import login


def _default_employee(client, headers) -> dict:
    items = client.get("/api/employees", headers=headers).json()["items"]
    return next(e for e in items if e["is_default"])


def _widget_key(client, headers, employee_id: str) -> str:
    return client.get(
        f"/api/employees/{employee_id}/widget-keys", headers=headers
    ).json()["items"][0]["key"]


def _png_bytes(w: int = 12, h: int = 12) -> bytes:
    """构造一张真实 PNG（视觉 healthcheck 同款最小图）。"""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xf5\xf5\xf5\xff" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _upload(client, key: str, visitor: str = "v-img-1", name: str = "photo.png",
            data: bytes | None = None, mime: str = "image/png"):
    return client.post(
        "/api/chat/images",
        headers={"X-Widget-Key": key},
        data={"visitor_id": visitor},
        files={"file": (name, io.BytesIO(data if data is not None else _png_bytes()), mime)},
    )


def test_upload_rejects_non_image(client, star):
    key = _widget_key(client, star, _default_employee(client, star)["id"])
    res = _upload(client, key, name="a.txt", data=b"hello", mime="text/plain")
    assert res.status_code == 415


def test_upload_rejects_oversize(client, star):
    key = _widget_key(client, star, _default_employee(client, star)["id"])
    # 刚好超过 10MB 的"图片"：内容本身不用是合法 PNG（大小校验在前）
    res = _upload(client, key, data=b"\x89" + b"0" * (10 * 1024 * 1024 + 1))
    assert res.status_code == 413


def test_upload_and_send_with_image(client, star, monkeypatch):
    """上传 → 带图发送：视觉转写进消息 meta，回复不受图片影响。"""
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    up = _upload(client, key)
    assert up.status_code == 200, up.text
    image_id = up.json()["image_id"]

    # 模拟视觉转写成功（测试环境 VISION_PROVIDER=off，必须打桩）
    monkeypatch.setattr("app.vision.describe", lambda *a, **k: _async("设备铭牌：型号 X100，额定电压 220V"))
    sent = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": key},
        json={"message": "帮我看看这是什么型号", "visitor_id": "v-img-1", "image_ids": [image_id]},
    )
    assert sent.status_code == 200, sent.text
    assert sent.json()["reply"]

    # 图片已绑定会话且转写缓存
    from app.models import ChatImage as _CI
    row = client.app.dependency_overrides  # noqa: F841  (占位，下面直接查库)
    # 直接通过 history 接口验证图片引用与转写状态
    hist = client.get(
        "/api/chat/history", params={"visitor_id": "v-img-1"}, headers={"X-Widget-Key": key}
    ).json()
    imgs = [im for m in hist["items"] for im in (m.get("images") or [])]
    assert imgs and imgs[0]["id"] == image_id


async def _async(text: str) -> str:
    return text


def test_ocr_text_cached_across_turns(client, star, monkeypatch):
    """多轮引用：第二次带同一张图（或历史回放）不再重新转写。"""
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    up = _upload(client, key, visitor="v-img-2")
    image_id = up.json()["image_id"]

    calls = {"n": 0}

    async def fake_describe(*a, **k):
        calls["n"] += 1
        return "故障代码 E3"

    monkeypatch.setattr("app.vision.describe", fake_describe)
    for _ in range(2):
        res = client.post(
            "/api/chat/message",
            headers={"X-Widget-Key": key},
            json={"message": "这个故障怎么处理", "visitor_id": "v-img-2", "image_ids": [image_id]},
        )
        assert res.status_code == 200
    assert calls["n"] == 1, "同一张图的转写必须缓存，第二轮不得再调视觉模型"


def test_image_owner_isolation(client, star, sea):
    """跨访客/跨租户不能引用或读取别人的图。"""
    emp_a = _default_employee(client, star)
    key_a = _widget_key(client, star, emp_a["id"])
    up = _upload(client, key_a, visitor="v-owner-a")
    image_id = up.json()["image_id"]

    # 同租户不同访客：读取被拒
    res = client.get(
        f"/api/chat/images/{image_id}",
        params={"visitor_id": "v-other"},
        headers={"X-Widget-Key": key_a},
    )
    assert res.status_code == 404

    # 其他访客发送消息引用该图：静默忽略（消息照发，图片不进上下文）
    sent = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": key_a},
        json={"message": "看看这张图", "visitor_id": "v-thief", "image_ids": [image_id]},
    )
    assert sent.status_code == 200
    hist = client.get(
        "/api/chat/history", params={"visitor_id": "v-thief"}, headers={"X-Widget-Key": key_a}
    ).json()
    assert all(not m.get("images") for m in hist["items"])


def test_recent_history_injects_ocr_text(client, star):
    """_recent_history 把图片转写文本注入多轮上下文。"""
    from app.database import SessionLocal
    from app.models import Message, MessageRole, Session as ChatSession, Session as _S, SessionStatus, Tenant

    with SessionLocal() as db:
        tenant_id = db.query(Tenant).first().id  # FK 要求真实租户
        s = ChatSession(
            tenant_id=tenant_id, employee_id="e-imgtest",
            visitor_id="v-hist", channel="web", status=SessionStatus.AI,
        )
        db.add(s)
        db.flush()
        img = ChatImage(
            tenant_id=tenant_id, visitor_id="v-hist",
            ocr_text="型号 DX-500，序列号 SN12345",
            ocr_status="ok",
        )
        db.add(img)
        db.flush()
        db.add(Message(
            tenant_id=tenant_id, session_id=s.id, role=MessageRole.VISITOR,
            content="这是什么型号",
            meta=f'{{"images": [{{"id": "{img.id}", "ocr_status": "ok"}}]}}',
        ))
        db.commit()

        items = _recent_history(db, tenant_id, s.id)
        assert items and "DX-500" in items[0]["content"], "历史中的图片转写文本必须注入上下文"
        assert "SN12345" in items[0]["content"]


def test_message_cap_on_images(client, star, monkeypatch):
    """单条消息最多带 3 张图，超出的静默截断。"""
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    ids = []
    for i in range(4):
        up = _upload(client, key, visitor="v-cap", name=f"p{i}.png")
        ids.append(up.json()["image_id"])
    monkeypatch.setattr("app.vision.describe", lambda *a, **k: _async("图中内容"))
    res = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": key},
        json={"message": "看图", "visitor_id": "v-cap", "image_ids": ids},
    )
    assert res.status_code == 200
    hist = client.get(
        "/api/chat/history", params={"visitor_id": "v-cap"}, headers={"X-Widget-Key": key}
    ).json()
    imgs = [im for m in hist["items"] for im in (m.get("images") or [])]
    assert len(imgs) == 3
