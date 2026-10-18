"""免登录模式（AUTH_DISABLED）行为测试。"""
from __future__ import annotations

from app.config import settings


def test_mode_reflects_flag(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_disabled", True)
    r = client.get("/api/auth/mode")
    assert r.status_code == 200
    assert r.json()["auth_disabled"] is True

    monkeypatch.setattr(settings, "auth_disabled", False)
    r = client.get("/api/auth/mode")
    assert r.json()["auth_disabled"] is False


def test_auto_login_disabled_returns_403(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_disabled", False)
    r = client.post("/api/auth/auto")
    assert r.status_code == 403


def test_auto_login_issues_tenant_admin_token(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_disabled", True)
    r = client.post("/api/auth/auto")
    assert r.status_code == 200
    data = r.json()
    assert data["access_token"]
    assert data["role"] == "tenant_admin"
    assert data["tenant_id"]
    # 拿令牌访问 me，验证身份链路完整
    me = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["tenant"]["id"] == data["tenant_id"]
