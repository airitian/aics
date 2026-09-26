"""共用的弹性原语：熔断器 + 可重试状态码判定。

向量库（Qdrant Cloud）和向量模型（Gitee AI）都在公网上，各自会遇到两类
**性质完全不同**的故障，处理策略也必须不同：

- **抖动**（TLS 被重置、连接超时）：退避重试几次就能吃掉，客户完全无感。
  对一条客户消息来说，重试的成本是几十毫秒，收益是「不用转人工」。
- **真故障**（集群暂停、Key 失效）：逐请求重试只会让**每条**消息都白等
  完整一轮退避还照样答不上来 —— 又慢又没用。这时要熔断：冷却窗口内
  立刻降级（快速给出兜底话术），冷却后再放一个探针请求过去。

两处的判定和阈值语义必须一致，所以抽到这里共用，避免改了一处漏了另一处。
"""
from __future__ import annotations

import threading
import time

# 上游/网关侧的错误码 —— 重试有意义
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_retryable_status(code: int | None) -> bool:
    return code in RETRYABLE_STATUS


class Breaker:
    """极简熔断器：连续失败到阈值后，在冷却窗口内直接失败，不再打网络。

    线程安全（向量库走线程、向量模型走协程，都会碰到）。
    """

    def __init__(self, threshold: int = 3, cooldown: float = 15.0):
        self.threshold = max(1, threshold)
        self.cooldown = max(0.0, cooldown)
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def on_success(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0

    def on_failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self._fails >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown

    def reset(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0
