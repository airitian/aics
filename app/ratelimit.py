"""per-tenant 限流与用量计量。

为什么必须按租户做（PRD 2.5 / 上线准备清单第 3、4 条）：
- 全平台共用一套模型端点，额度是共享的。不做租户级限流，一个租户的洪峰会把
  额度吃光，直接违反「单租户超限不得影响其他租户」。
- 模型成本由平台承担，没有租户级计量就无法配额、无法定价、无法防滥用。

注意：内置限流器是**进程内**的。多 worker / 多实例部署时请换成 Redis 实现
（只需替换 TenantRateLimiter 的 acquire/release，调用方不用改）。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Tenant, UsageRecord
from app.utils import utcnow


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int = 1):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class QuotaExceeded(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TenantRateLimiter:
    """滑动窗口 RPM + 在途并发，均按租户隔离。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._inflight: dict[str, int] = {}

    def acquire(self, tenant_id: str, rpm: int, concurrent: int) -> None:
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(tenant_id, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= max(1, rpm):
                raise RateLimited(
                    f"当前租户请求过于频繁（上限 {rpm} 次/分钟），请稍后重试", retry_after=2
                )
            inflight = self._inflight.get(tenant_id, 0)
            if inflight >= max(1, concurrent):
                raise RateLimited(
                    f"当前租户并发会话已达上限（{concurrent}），请稍后重试", retry_after=1
                )
            window.append(now)
            self._inflight[tenant_id] = inflight + 1

    def release(self, tenant_id: str) -> None:
        with self._lock:
            left = self._inflight.get(tenant_id, 0) - 1
            if left <= 0:
                self._inflight.pop(tenant_id, None)
            else:
                self._inflight[tenant_id] = left

    @contextmanager
    def slot(self, tenant_id: str, rpm: int, concurrent: int) -> Iterator[None]:
        self.acquire(tenant_id, rpm, concurrent)
        try:
            yield
        finally:
            self.release(tenant_id)


limiter = TenantRateLimiter()


# --------------------------------------------------------------------------- #
# 用量与配额
# --------------------------------------------------------------------------- #
def _today_start():
    return utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def tokens_used_today(db: Session, tenant_id: str) -> int:
    stmt = select(func.coalesce(func.sum(UsageRecord.total_tokens), 0)).where(
        UsageRecord.tenant_id == tenant_id, UsageRecord.created_at >= _today_start()
    )
    return int(db.execute(stmt).scalar_one() or 0)


def ensure_token_quota(db: Session, tenant: Tenant, estimated: int = 0) -> None:
    used = tokens_used_today(db, tenant.id)
    if used + estimated > tenant.daily_token_quota:
        raise QuotaExceeded(
            f"本租户今日模型额度已用尽（{used}/{tenant.daily_token_quota}），"
            "AI 接待已降级，请联系管理员调整配额"
        )


def record_usage(
    db: Session,
    *,
    tenant_id: str,
    employee_id: str | None,
    kind: str = "llm",
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    origin: str = "chat",
    session_id: str | None = None,
    commit: bool = False,
) -> UsageRecord:
    rec = UsageRecord(
        tenant_id=tenant_id,
        employee_id=employee_id,
        kind=kind,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        origin=origin,
        session_id=session_id,
    )
    db.add(rec)
    if commit:
        db.commit()
    return rec
