"""通用小工具。"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """统一使用 naive UTC，兼容 SQLite 与 PostgreSQL。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def j_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def j_load(raw: str | None, default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if text is None:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def count_chars(text: str) -> int:
    """PRD 3.1.3 计数口径：中英文、标点、空格、换行各计 1。"""
    return len(text or "")


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars(text: str) -> str:
    """过滤不可见控制字符（保留 \t \n \r）。"""
    return _CONTROL_RE.sub("", text or "")


def mask_secret(value: str | None, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 8}{value[-keep:]}"


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", (text or "")).strip()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", (name or "").strip().lower())
    return s.strip("-") or new_id()[:8]
