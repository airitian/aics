"""测试夹具：独立数据库 + 演示租户。"""
from __future__ import annotations

import os
from pathlib import Path

# 必须在导入 app 之前设置环境变量
_TEST_DB = Path(__file__).parent / "test_aics.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
os.environ["ENV"] = "test"
os.environ["BOOTSTRAP_DEMO"] = "true"
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = "admin@aics.local"
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = "admin12345"
os.environ["LLM_PROVIDER"] = "stub"
os.environ["EMBED_PROVIDER"] = "local"
os.environ["VECTOR_BACKEND"] = "db"
os.environ["MIN_SCORE"] = "0.05"
os.environ["JWT_SECRET"] = "test-only-secret-key-not-for-production-0f3c9d7a21b8"
# 测试必须离线可跑：视觉/重排一旦跟随 .env 开到 api，
# 任何带图文档的上传用例都会真打外网（慢、烧钱、还会随机失败）
os.environ["VISION_PROVIDER"] = "off"
os.environ["RERANK_PROVIDER"] = "off"
# 上传落盘写到测试专用目录，别把测试数据写进真实 assets/
os.environ["ASSETS_DIR"] = str((Path(__file__).parent / ".tmp_assets").as_posix())

if _TEST_DB.exists():
    _TEST_DB.unlink()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def login(client: TestClient, email: str, password: str) -> dict:
    res = client.post("/api/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200, res.text
    return {"Authorization": "Bearer " + res.json()["access_token"]}


@pytest.fixture(scope="session")
def star(client):
    """演示租户 A（星辰户外装备）管理员。"""
    return login(client, "star@demo.local", "demo12345")


@pytest.fixture(scope="session")
def sea(client):
    """演示租户 B（海蓝家居）管理员。"""
    return login(client, "sea@demo.local", "demo12345")


@pytest.fixture(scope="session")
def platform(client):
    """平台管理员。"""
    return login(client, "admin@aics.local", "admin12345")
