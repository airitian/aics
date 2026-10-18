"""租户隔离与权限测试 —— 本文件是第二章的验收标准落地。"""
from __future__ import annotations

import io

import pytest

from tests.conftest import login


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def test_health(client):
    assert client.get("/healthz").json()["ok"] is True


def test_meta_exposes_limits(client):
    data = client.get("/api/meta").json()
    assert data["limits"]["persona_max_chars"] == 20000
    assert data["limits"]["max_employees_per_tenant"] == 10


def test_platform_admin_has_no_tenant_context(client, platform):
    """平台账号不得以租户身份操作租户级接口。"""
    res = client.get("/api/employees", headers=platform)
    assert res.status_code == 403
    assert "租户上下文" in res.json()["detail"]


def test_tenant_user_cannot_use_platform_api(client, star):
    res = client.get("/api/platform/tenants", headers=star)
    assert res.status_code == 403


def test_missing_token_rejected(client):
    assert client.get("/api/employees").status_code == 401


# --------------------------------------------------------------------------- #
# 跨租户读取：必须查不到，且不泄露存在性
# --------------------------------------------------------------------------- #
def test_employee_of_other_tenant_is_not_readable(client, star, sea):
    star_emp = client.get("/api/employees", headers=star).json()["items"][0]["id"]
    res = client.get(f"/api/employees/{star_emp}", headers=sea)
    assert res.status_code == 404, "跨租户读取应返回 404（不区分不存在与无权限）"

    res = client.patch(f"/api/employees/{star_emp}", headers=sea, json={"name": "被篡改"})
    assert res.status_code == 404

    res = client.delete(f"/api/employees/{star_emp}", headers=sea)
    assert res.status_code == 404


def test_kb_of_other_tenant_is_not_readable(client, star, sea):
    star_kb = client.get("/api/knowledge-bases", headers=star).json()["items"][0]["id"]
    assert client.get(f"/api/knowledge-bases/{star_kb}/documents", headers=sea).status_code == 404
    assert client.delete(f"/api/knowledge-bases/{star_kb}", headers=sea).status_code == 404


def test_cannot_bind_other_tenant_kb(client, sea, star):
    """把别人的知识库挂到自己的 AI 员工上 —— 必须被拒。"""
    star_kb = client.get("/api/knowledge-bases", headers=star).json()["items"][0]["id"]
    sea_emp = client.get("/api/employees", headers=sea).json()["items"][0]["id"]
    res = client.patch(f"/api/employees/{sea_emp}", headers=sea, json={"kb_ids": [star_kb]})
    assert res.status_code == 400
    assert "不属于本租户" in res.json()["detail"]


# --------------------------------------------------------------------------- #
# 检索层隔离：同样的问法，只能命中本租户的资料
# --------------------------------------------------------------------------- #
def test_retrieval_is_tenant_scoped(client, star, sea):
    """星晨的检索里绝不能出现海蓝的专属信息，反之亦然。"""
    res = client.post(
        "/api/knowledge/search-test",
        headers=star,
        json={"query": "海蓝家居 大客户 工程单 返点 SEA30", "top_k": 5},
    )
    assert res.status_code == 200
    items = res.json()["items"]
    for hit in items:
        assert "SEA30" not in hit["text"]

    res = client.post(
        "/api/knowledge/search-test",
        headers=sea,
        json={"query": "星辰户外 老客户 专属折扣码 STAR20", "top_k": 5},
    )
    assert res.status_code == 200
    for hit in res.json()["items"]:
        assert "STAR20" not in hit["text"]


# --------------------------------------------------------------------------- #
# 渠道凭证：租户由凭证绑定关系推导
# --------------------------------------------------------------------------- #
def test_widget_key_invalid(client):
    res = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": "wk_not_exists"},
        json={"message": "你好", "visitor_id": "v1"},
    )
    assert res.status_code == 401


def test_widget_key_derives_tenant_and_isolates(client, star, sea):
    star_key = client.get("/api/employees", headers=star).json()["items"][0]["id"]
    keys = client.get(f"/api/employees/{star_key}/widget-keys", headers=star).json()["items"]
    assert keys, "默认应该带一个网页渠道凭证"
    key = keys[0]["key"]

    cfg = client.get("/api/chat/widget-config", headers={"X-Widget-Key": key}).json()
    assert cfg["tenant_name"] == "星辰户外装备"

    # 拿星辰的凭证问海蓝的专属信息 → 不得命中
    res = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": key},
        json={"message": "你们的大客户工程单返点比例 SEA30 是多少？", "visitor_id": "v-isolation"},
    )
    assert res.status_code == 200
    body = res.json()
    for hit in body["hits"]:
        assert "SEA30" not in hit["text"]


def test_chat_session_not_visible_to_other_tenant(client, star, sea):
    star_emp = client.get("/api/employees", headers=star).json()["items"][0]["id"]
    key = client.get(f"/api/employees/{star_emp}/widget-keys", headers=star).json()["items"][0]["key"]
    sid = client.post(
        "/api/chat/message",
        headers={"X-Widget-Key": key},
        json={"message": "你们的帐篷防水多少？", "visitor_id": "v-sess"},
    ).json()["session_id"]

    assert client.get(f"/api/sessions/{sid}/messages", headers=sea).status_code == 404
    assert client.post(f"/api/sessions/{sid}/takeover", headers=sea).status_code == 404


# --------------------------------------------------------------------------- #
# 租户停用：只影响本租户
# --------------------------------------------------------------------------- #
def test_suspended_tenant_is_locked_out(client, platform, star, sea):
    tenants = client.get("/api/platform/tenants", headers=platform).json()["items"]
    star_tenant = next(t for t in tenants if t["name"] == "星辰户外装备")
    try:
        res = client.patch(
            f"/api/platform/tenants/{star_tenant['id']}", headers=platform, json={"status": "suspended"}
        )
        assert res.status_code == 200
        assert client.get("/api/employees", headers=star).status_code == 403
        # 另一个租户不受影响
        assert client.get("/api/employees", headers=sea).status_code == 200
    finally:
        client.patch(
            f"/api/platform/tenants/{star_tenant['id']}", headers=platform, json={"status": "active"}
        )


def test_login_blocked_for_suspended_tenant(client, platform):
    tenants = client.get("/api/platform/tenants", headers=platform).json()["items"]
    target = next(t for t in tenants if t["name"] == "海蓝家居")
    client.patch(f"/api/platform/tenants/{target['id']}", headers=platform, json={"status": "suspended"})
    try:
        res = client.post("/api/auth/login", json={"email": "sea@demo.local", "password": "demo12345"})
        assert res.status_code == 403
    finally:
        client.patch(f"/api/platform/tenants/{target['id']}", headers=platform, json={"status": "active"})


# --------------------------------------------------------------------------- #
# 平台侧
# --------------------------------------------------------------------------- #
def test_platform_tenant_lifecycle(client, platform):
    res = client.post(
        "/api/platform/tenants",
        headers=platform,
        json={
            "name": "测试租户",
            "admin_email": "t1@demo.local",
            "admin_password": "demo12345",
        },
    )
    assert res.status_code == 200
    tenant_id = res.json()["id"]

    # 新租户开箱即有默认 AI 员工 + 知识库 + 渠道凭证
    headers = login(client, "t1@demo.local", "demo12345")
    employees = client.get("/api/employees", headers=headers).json()["items"]
    assert len(employees) == 1
    assert employees[0]["is_default"] is True
    assert len(employees[0]["kb_ids"]) == 1

    keys = client.get(f"/api/employees/{employees[0]['id']}/widget-keys", headers=headers).json()["items"]
    assert len(keys) == 1

    # 注销删除需要确认标识一致
    slug = next(t["slug"] for t in client.get("/api/platform/tenants", headers=platform).json()["items"] if t["id"] == tenant_id)
    assert client.post(f"/api/platform/tenants/{tenant_id}/purge?confirm=wrong", headers=platform).status_code == 400
    assert client.post(f"/api/platform/tenants/{tenant_id}/purge?confirm={slug}", headers=platform).status_code == 200

    # 删除后该租户的数据不可再访问
    assert client.post("/api/auth/login", json={"email": "t1@demo.local", "password": "demo12345"}).status_code == 401


def test_upload_and_parse_error_is_actionable(client, star):
    kb = client.get("/api/knowledge-bases", headers=star).json()["items"][0]["id"]

    # 不支持的格式
    res = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={"files": ("bad.exe", b"binary", "application/octet-stream")},
    )
    item = res.json()["items"][0]
    assert item["status"] == "failed"
    assert "不支持" in item["error"]

    # 空文件
    res = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={"files": ("empty.txt", b"", "text/plain")},
    )
    assert res.json()["items"][0]["status"] == "failed"

    # 正常入库
    text = "本店所有商品支持 30 天价保，价保期内降价可申请退还差价。".encode()
    res = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={"files": ("价保政策.txt", text, "text/plain")},
    )
    item = res.json()["items"][0]
    assert item["status"] == "success"
    assert item["chunks"] >= 1


def test_duplicate_upload_is_rejected(client, star):
    """同一份文件重复上传必须拦掉 —— 重复副本会**同分**挤满检索召回位。"""
    kb = client.get("/api/knowledge-bases", headers=star).json()["items"][0]["id"]
    payload = "本店支持 7 天无理由退货，需保持商品完好。".encode("utf-8")

    first = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={"files": ("退货政策.txt", payload, "text/plain")},
    ).json()
    assert first["items"][0]["status"] == "success", first
    assert first["accepted"] == 1

    # 同名 + 同大小 → 拒绝
    second = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={"files": ("退货政策.txt", payload, "text/plain")},
    ).json()
    item = second["items"][0]
    assert item["status"] == "failed", second
    assert "已在库中" in item["error"], item
    assert second["accepted"] == 0

    # 同一次请求里传两遍也要拦住（靠前面的 flush 生效）
    third = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files=[
            ("files", ("同批A.txt", b"AAAA", "text/plain")),
            ("files", ("同批A.txt", b"AAAA", "text/plain")),
        ],
    ).json()
    assert third["accepted"] == 1, third
    assert [i["status"] for i in third["items"]] == ["success", "failed"], third

    # 同名但内容不同（大小不同）不能被误伤
    diff = client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=star,
        files={
            "files": (
                "退货政策.txt",
                "本店支持 15 天无理由退货，需保持商品完好并附带发票。".encode("utf-8"),
                "text/plain",
            )
        },
    ).json()
    assert diff["items"][0]["status"] == "success", diff
