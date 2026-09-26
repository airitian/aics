"""认证与凭证。

硬规则（PRD 2.3）：租户标识只能由服务端推导 ——
本模块的 JWT 把 tid 签进令牌，请求侧永远不读 body/query 里的 tenant_id。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import timedelta

import jwt

from app.config import settings
from app.utils import utcnow

_PBKDF2_ITER = 260_000
_ALGO_TAG = "pbkdf2_sha256"


# --------------------------------------------------------------------------- #
# 口令
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITER)
    return f"{_ALGO_TAG}${_PBKDF2_ITER}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != _ALGO_TAG:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return hmac.compare_digest(expected, actual)
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# 令牌
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Principal:
    user_id: str
    email: str
    role: str
    tenant_id: str | None  # 平台运营为 None

    @property
    def is_platform(self) -> bool:
        return self.role == "platform_admin"


def create_access_token(*, user_id: str, email: str, role: str, tenant_id: str | None) -> str:
    now = utcnow()
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "tid": tenant_id or "",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def decode_access_token(token: str) -> Principal | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except jwt.PyJWTError:
        return None
    return Principal(
        user_id=str(payload.get("sub") or ""),
        email=str(payload.get("email") or ""),
        role=str(payload.get("role") or ""),
        tenant_id=(payload.get("tid") or None),
    )


def new_widget_key(prefix: str = "wk") -> str:
    return f"{prefix}_{secrets.token_urlsafe(24)}"
