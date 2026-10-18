"""单轮对话的分阶段耗时统计。

运维排查「回复慢」时能看到的只有端到端 latency（`routers/testchat.py` 里的
`time.perf_counter()` 差值），看不到时间究竟花在哪一环：是 embedding 模型慢、
向量库查询慢、重排慢，还是 LLM 生成慢。本模块提供**零依赖的轻量埋点**，
让每个阶段各自计时，轮次结束时打一行 `key=value` 摘要进日志。

设计要点（都是踩过坑才定下来的）：

1. **为什么用 `contextvars` 而不是层层传参**：`rag.retrieve()` 与 `agent.handle_turn`
   之间没有任何参数传递窗口（`retrieve` 是 await 出去的，embedding / rerank 的耗时
   只在 `rag.py` / `rag_llama.py` 内部可见）。要在那一层埋点就得改函数签名并一路传下去，
   而 `contextvars.ContextVar` 天然随 async 任务隔离，是这里最干净的做法。
   顺带白拿一个好处：**并发请求天然不互相污染**（每个请求有自己的 context）。

   注意：llama 引擎里 `QueryFusionRetriever` 用 `asyncio.gather` 起 Task，
   gather 创建的 Task 会**复制**当前 context（拿到的是同一个 collector 对象引用），
   所以在 `_retrieve` 里埋点照样能写回同一份数据。

2. **计时器里不做字符串格式化**：每轮对话会调用几十次 `stage()`，累积的格式化开销
   与无谓分配都是白付。计时器只做 `perf_counter()` 差值累加（float），
   字符串拼装一律推迟到 `log_summary()` 这唯一一次输出。

3. **异常时不丢已耗时**：阶段里抛异常（模型超时、向量库 500）恰恰是最需要看到
   「卡在哪一环」的时候，所以 `stage()` 用 `try/finally` 收尾。

4. **同一阶段可重复出现**：LLM 有三条调用路径（主生成 / 上下文兜底 / 概览兜底），
   embedding 在概览兜底里会调第二次。同名阶段**耗时累加、次数累加**，日志里同时给出
   `llm=6235ms` 与 `llm_calls=2`，避免「值看着很小其实是调了两次」的误读。

**关于口径：只有 `total` 是包含式的**，各阶段之间**不可相加**。
这是本模块最容易被误读的一点，所以必须写死：

- `total` 覆盖整轮（含未归因的框架开销），是唯一的真超集。
- `fusion` **不是** `vector_search` / `bm25` 的超集。llama 引擎里这两路各有**两个**
  埋点：「建连探测 + 建索引」在 fusion **之外**（retrieve 里那段），
  「查询 + 检索」在 fusion **之内**（检索器自己的 `_retrieve`）。
  所以会出现 `fusion=1098ms` 而 `vector_search=5512ms` —— 这是真实结构，不是 bug。
- `overview` 是它内部那次二次检索（embed / vector_search）的超集。

正因为「各阶段之和 ≠ total」，**不要拿各阶段相加去做容量规划**：
要判断「检索占了多少」，看 `total` 与各阶段的量级关系；要判断「哪一环慢」，
看单个阶段的值。加总会重复计入嵌套部分。

阶段 key 清单见 `STAGE_KEYS`，日志格式见 `format_summary`。
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

# 全部阶段 key（新增埋点时同步登记，保证日志字段顺序稳定、可直接 grep/diff）。
#
# 顺序即日志里的输出顺序，按「链路自上而下」排：编排判断 → 检索 → 历史/提示词 → 生成。
# 固定顺序不是为了好看，而是让运维能直接 `grep '链路耗时'` 之后横向对比多轮日志，
# 字段跳来跳去会让肉眼比对失效。
STAGE_KEYS: tuple[str, ...] = (
    "intent",         # 意图与风险判定（纯 CPU）
    "webfetch",       # 网页内容提取：访客发来链接时抓正文（外部 HTTP，可并发多链接）
    "kb_scope",       # 取员工绑定的知识库/文件范围（查库）
    "embed",          # embedding 模型远程调用（rag / rag_llama 内部）
    "vector_search",  # 向量库查询（Qdrant Cloud 公网）
    "bm25",           # BM25 稀疏召回（仅 llama 引擎）
    "fusion",         # RRF 融合，含上面两路（仅 llama 引擎）
    "rerank",         # 重排模型远程调用
    "history",        # 读会话历史 / 跨会话历史 / 访客记忆（查库）
    "prompt_build",   # 拼系统提示词与消息列表（纯 CPU）
    "llm",            # LLM 生成（远程调用，可能多次）
    "overview",       # 概览兜底素材准备（查库 + 二次检索；含二次 embed/vector）
)


@dataclass
class TurnTiming:
    """一轮对话的耗时收集器。

    字段刻意保持「只存数字 + 极少量定位信息」：耗时是 float 毫秒、次数是 int，
    任何展示用的格式化都留到 `format_summary()`。
    """

    # 端到端起点（perf_counter 秒）。由 `turn()` 在进入时立刻取，
    # 所以 total 覆盖的是「这一轮对话」的全部时间，不含计时器自身的准备开销。
    started: float = field(default_factory=time.perf_counter)
    # 定位前缀，形如 `tenant=xxx emp=yyy sess=zzz origin=chat`。
    # 只存字符串、不做拼装以外的事；空值由 format_summary 跳过。
    prefix: str = ""
    # 最终走的分支（如 `normal` / `handoff_lowconf`），由 agent 的每个 return 点登记。
    path: str = ""
    # 阶段 key → 累计耗时（毫秒）。同一 key 多次出现则累加。
    stage_ms: dict[str, float] = field(default_factory=dict)
    # 阶段 key → 出现次数。用于回答「这个值是调一次还是调两次」。
    stage_calls: dict[str, int] = field(default_factory=dict)

    _total_ms: float | None = field(default=None, repr=False)

    # ---- 计时侧（热路径，只做数值累加） ----

    def record(self, name: str, elapsed_ms: float) -> None:
        """累加一次阶段耗时（`elapsed_ms` 已由 `stage()` 换算好）。"""
        self.stage_ms[name] = self.stage_ms.get(name, 0.0) + elapsed_ms
        self.stage_calls[name] = self.stage_calls.get(name, 0) + 1

    def total_ms(self) -> float:
        """端到端耗时（毫秒）。

        **首次读取时定格**：真正「这一轮结束了」的时刻就是第一次打摘要的时刻，
        之后再读不会让total 一直往上涨（否则同一轮打印两次会得到两个不同的 total，
        对不上账）。
        """
        if self._total_ms is None:
            self._total_ms = (time.perf_counter() - self.started) * 1000.0
        return self._total_ms

    # ---- 查询侧（冷路径，测试与排障用） ----

    def count(self, name: str) -> int:
        return self.stage_calls.get(name, 0)

    def ms(self, name: str) -> float:
        return self.stage_ms.get(name, 0.0)

    def as_dict(self) -> dict[str, float]:
        """按 `STAGE_KEYS` 的固定顺序导出耗时（毫秒），便于测试断言。"""
        return {k: round(self.stage_ms[k], 3) for k in STAGE_KEYS if k in self.stage_ms}


# 当前轮次的收集器。default=None 表示「不在一轮对话里」（如知识库测试接口直接调
# rag.retrieve、入库流程、单测）—— 此时 `stage()` 退化成空上下文管理器，
# 不产生任何计时与副作用。
_current: ContextVar[TurnTiming | None] = ContextVar("aics_turn_timing", default=None)


def current() -> TurnTiming | None:
    """取当前轮次的收集器；不在一轮对话里时返回 None。"""
    return _current.get()


@contextmanager
def turn(prefix: str = "") -> Iterator[TurnTiming]:
    """开启一轮计时。

    退出时用 `reset(token)` 还原而不是 `set(None)`：嵌套开启时内层结束后能正确回到
    外层的收集器，不会把外层顶掉（`stage()` 支持嵌套正是靠这一条）。
    """
    collector = TurnTiming(prefix=prefix)
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """给一段代码计时，累加到当前轮次的 `name` 阶段上。

    支持嵌套（llama 引擎在 `retrieve` 内部再嵌一层）与同名重复（LLM 三条路径）。
    异常路径用 `finally` 收尾：**已经花掉的时间必须留下**——
    「模型超时」这件事本身就是最需要耗时信息的场景。

    不在任何轮次里时（`current()` 为 None）直接透传，连`perf_counter()` 都不调。
    """
    collector = _current.get()
    if collector is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        # 这里保留完整浮点精度（不提前取整）：纯 CPU 阶段常在 0.01~0.99ms 之间，
        # 中途取整会把它们压成 0，于是「有耗时但很快」和「压根没跑」分不出来。
        # 展示精度由 `_fmt_ms` 在最终输出那一次决定。
        collector.record(name, (time.perf_counter() - start) * 1000.0)


def kv(key: str, value: object) -> str:
    """拼一个 `key=value` 字段，值里的空格替换成 `_`。

    日志字段一律不许含裸空格 —— 否则 `grep '链路耗时' | cut -d' '` 切列会错位，
    采集侧也提不出结构化指标。调用方用它拼 `prefix`（tenant/emp/sess/origin）。
    """
    if value is None or value == "":
        return ""
    return f"{key}={str(value).replace(' ', '_')}"


def _fmt_ms(value: float) -> str:
    """耗时格式化：整数直接输出（`12ms`），带小数才保留一位（`0.4ms`）。

    避免出现 `12.0ms` 这种冗余尾巴，grep 时也更干净。
    """
    rounded = round(value, 1)
    if rounded == int(rounded):
        return f"{int(rounded)}ms"
    return f"{rounded}ms"


def format_summary(
    collector: TurnTiming,
    *,
    tokens: int = 0,
    degraded: bool = False,
    **extra: object,
) -> str:
    """拼出单行摘要（不含时间戳/level，那些由日志框架拼）。

    格式::

        链路耗时 tenant=x emp=y sess=z origin=chat total=12450ms 口径=total_incl 阶段: \
intent=1ms ... llm=6235ms embed_calls=1 llm_calls=1 tokens=1234 degraded=0 path=normal

    `口径=total_incl` 是给**看日志的人**看的，不是给读源码的人看的：
    它声明「只有 total 是包含式超集，阶段之间不可相加」。
    没有这个标记时，运维看到 `fusion=1098ms vector_search=5512ms` 会怀疑数据坏了，
    或者更糟——把各阶段相加去做容量规划，重复计入嵌套部分后得出错误结论。

    `prefix` 原样透传（它已由 `kv()` 逐字段清洗过），其余字段用 `kv()` 拼。
    全部字段用 `key=value`、值里不出现空格，这样 `grep '链路耗时'` 之后可以逐字段解析，
    也能被日志采集侧正则提成结构化指标。

    `path` 是最终走的分支、`degraded` 标降级——排查「为什么这次是兜底话术」时，
    这两个字段和耗时同样重要。
    """
    parts: list[str] = ["链路耗时"]
    if collector.prefix:
        parts.append(collector.prefix)
    parts.append(kv("total", _fmt_ms(collector.total_ms())))
    parts.append(kv("口径", "total_incl"))
    ordered = [k for k in STAGE_KEYS if k in collector.stage_ms]
    # 兜住 STAGE_KEYS 里没登记的 key（临时试点的阶段），避免埋点写了却打不出来
    ordered += [k for k in collector.stage_ms if k not in STAGE_KEYS]
    if ordered:
        parts.append("阶段: " + " ".join(kv(k, _fmt_ms(collector.stage_ms[k])) for k in ordered))
        parts.append(" ".join(kv(f"{k}_calls", collector.stage_calls[k]) for k in ordered))
    else:
        # 显式写明「一个阶段都没跑」，而不是留空。渠道未启用 / 会话已停止这类
        # 分支会在任何埋点之前就return，留空会被读成「阶段都很快」，
        # 实际上是没走到任何有开销的环节 —— 两者的排障结论完全不同。
        parts.append("阶段: 无")
    parts.extend(k for k in (kv(key, value) for key, value in extra.items()) if k)
    if tokens:
        parts.append(kv("tokens", tokens))
    parts.append(kv("degraded", 1 if degraded else 0))
    if collector.path:
        parts.append(kv("path", collector.path))
    return " ".join(p for p in parts if p)


def log_summary(
    logger: logging.Logger,
    collector: TurnTiming | None,
    *,
    tokens: int = 0,
    degraded: bool = False,
    level: int = logging.INFO,
    **extra: object,
) -> str:
    """输出耗时摘要，返回这行文本（测试可直接断言返回值，不必去抓 logging handler）。

    `collector` 为 None（不在一轮对话里）时静默返回空串——保证埋点可以被安全地
    加在那些「有时在一轮对话里、有时不是」的公共函数上（如 `rag.retrieve`）。
    """
    if collector is None:
        return ""
    line = format_summary(collector, tokens=tokens, degraded=degraded, **extra)
    logger.log(level, "%s", line)
    return line