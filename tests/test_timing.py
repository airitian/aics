"""每轮对话的分阶段耗时统计（app/timing.py）。

埋点是纯观测改动，最大的风险不是「计时不准」，而是**顺手改掉了业务行为**：
多一句 `result.path = ...` 就可能覆盖掉 `result.reply`，多包一层 `with` 就可能
把异常吞掉。所以这里一半的用例在钉计时器本身，另一半在钉**行为不变**。

覆盖的五件事：
1. `stage()` 基本计时 / 嵌套 / 同名累加；
2. 异常时不丢已耗时（模型超时恰恰最需要知道卡在哪一环）；
3. `contextvars` 在 async 下按请求隔离（并发不串味）；
4. 一轮 `handle_turn` 打出含全部阶段 key 的**单行**摘要；
5. 回归：埋点前后的回复内容、handoff 标志、tokens 完全一致。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from unittest import mock

import pytest

from app import timing
from app.timing import TurnTiming, format_summary, log_summary, stage, turn

from tests.test_flows import _default_employee, _widget_key


# --------------------------------------------------------------------------- #
# stage()：基本计时 / 嵌套 / 累加
# --------------------------------------------------------------------------- #
def test_stage_records_elapsed_time():
    with turn() as collector:
        with stage("llm"):
            time.sleep(0.02)
    assert collector.count("llm") == 1
    # sleep(20ms) 的实测耗时应落在 20~200ms；下界给足余量防 CI 抖动，上界防「计时单位错了」
    assert 15.0 <= collector.ms("llm") <= 200.0, f"llm 耗时不合理：{collector.ms('llm')}ms"


def test_stage_outside_turn_is_noop():
    """不在一轮对话里（如知识库测试接口直接调 rag.retrieve）时埋点必须是空操作。

    这是埋点能安全加在公共函数上的前提 —— 不能因为加了计时就把接口搞崩。
    """
    with stage("embed"):
        pass
    assert timing.current() is None


def test_nested_stages_are_independent():
    """嵌套时内层只算自己的，外层算包含内层的整段。"""
    with turn() as collector:
        with stage("fusion"):
            with stage("vector_search"):
                time.sleep(0.02)
            with stage("bm25"):
                time.sleep(0.02)
    assert collector.count("fusion") == 1
    assert collector.count("vector_search") == 1
    assert collector.count("bm25") == 1
    # 外层必须 ≥ 两个内层之和（外层含内层），否则嵌套语义就错了
    inner = collector.ms("vector_search") + collector.ms("bm25")
    assert collector.ms("fusion") >= inner, (
        f"外层 {collector.ms('fusion')}ms 应包含内层合计 {inner}ms"
    )


def test_same_stage_called_twice_accumulates():
    """LLM 有三条调用路径，同名阶段必须「耗时累加 + 次数累加」。

    只记最后一次的话，`llm=30ms llm_calls=2` 会变成 `llm=15ms`，
    让人以为模型很快 —— 次数是这个字段存在的全部意义。
    """
    with turn() as collector:
        for _ in range(3):
            with stage("llm"):
                time.sleep(0.01)
    assert collector.count("llm") == 3
    assert collector.ms("llm") >= 25.0, f"三次耗时未累加：{collector.ms('llm')}ms"


def test_stage_keeps_elapsed_on_exception():
    """阶段里抛异常时，已经花掉的时间必须留下。

    「模型超时 30 秒」这件事本身就是最需要耗时信息的场景，
    用 except 吞掉计时就等于在最该有数据的时候没有数据。
    """
    with turn() as collector:
        with pytest.raises(RuntimeError, match="boom"):
            with stage("llm"):
                time.sleep(0.02)
                raise RuntimeError("boom")
    assert collector.count("llm") == 1
    assert collector.ms("llm") >= 15.0, f"异常路径丢失了耗时：{collector.ms('llm')}ms"


def test_turn_restores_outer_collector():
    """内层 turn 结束后必须回到外层的收集器，不能把外层顶掉。"""
    with turn() as outer:
        with turn() as inner:
            with stage("inner_only"):
                pass
        assert timing.current() is outer, "内层结束后应恢复外层收集器"
        with stage("outer_only"):
            pass
    assert outer.count("outer_only") == 1
    assert outer.count("inner_only") == 0, "内层的数据不该串到外层"
    assert inner.count("inner_only") == 1


# --------------------------------------------------------------------------- #
# contextvars：async 下按请求隔离
# --------------------------------------------------------------------------- #
def test_concurrent_turns_do_not_leak():
    """两个并发「请求」各自计时，互不污染。

    contextvars 在 asyncio 里每创建一个 Task 就拷贝一份当前 context，
    拷贝的是**引用**，所以两个 Task 拿到的是同一个 collector 对象 ——
    这意味着如果收集器被共享，并发请求的阶段数据就会互相覆盖。
    这里用「谁在睡」把时序拉开，确保真的交错执行过。
    """
    seen: dict[str, dict[str, float]] = {}

    async def fake_turn(name: str, delay: float) -> TurnTiming:
        with turn(prefix=f"tenant={name}") as collector:
            with stage(name):
                await asyncio.sleep(delay)
            # 用 stage_ms 而不是 as_dict()：as_dict 只导出 STAGE_KEYS 里的 key，
            # 这里故意用非标准阶段名来区分两个并发请求。
            seen[name] = dict(collector.stage_ms)
            # 交回控制权时，另一个 Task 应该已经跑起来了
            await asyncio.sleep(0)
            return collector

    async def main():
        return await asyncio.gather(
            fake_turn("alpha", 0.05), fake_turn("beta", 0.01)
        )

    first, second = asyncio.run(main())

    # 每个 Task 只该看到自己那一个阶段
    assert set(seen["alpha"]) == {"alpha"}, f"alpha 的数据被污染：{seen['alpha']}"
    assert set(seen["beta"]) == {"beta"}, f"beta 的数据被污染：{seen['beta']}"
    assert first.count("alpha") == 1 and first.count("beta") == 0
    assert second.count("beta") == 1 and second.count("alpha") == 0
    assert first.prefix == "tenant=alpha" and second.prefix == "tenant=beta"


def test_collector_not_visible_after_turn():
    """一轮结束后收集器必须从 context 里摘掉。

    不摘的话：同一个 worker 线程处理下一个请求时，任何残留的 stage() 埋点
    都会写到上一轮的数据里，耗时越攒越离谱。
    """
    with turn():
        with stage("llm"):
            pass
    assert timing.current() is None


# --------------------------------------------------------------------------- #
# 摘要格式
# --------------------------------------------------------------------------- #
def test_summary_is_single_line_and_greppable():
    """摘要必须是**单行** `key=value`，字段顺序稳定，可直接 grep / 逐字段解析。"""
    with turn(prefix="tenant=t1 emp=e1 sess=s1 origin=chat") as collector:
        with stage("embed"):
            pass
        with stage("llm"):
            pass
        with stage("llm"):
            pass
    collector.path = "normal"
    line = format_summary(collector, tokens=1234, degraded=False)

    assert "\n" not in line, "摘要必须是单行"
    assert line.startswith("链路耗时 tenant=t1 emp=e1 sess=s1 origin=chat total=")
    for expect in ("embed=", "llm=", "embed_calls=1", "llm_calls=2",
                   "tokens=1234", "degraded=0", "path=normal"):
        assert expect in line, f"摘要缺字段 {expect}：{line}"
    # 阶段段内部的 token 不能出现裸空格（`embed=0.4ms llm=1.2ms`），
    # 否则按空格切列会与 key=value 混在一起，采集侧无法稳定解析
    stage_part = line.split(" 阶段: ")[1].split(" embed_calls=")[0]
    assert all(" " not in tok and "=" in tok for tok in stage_part.split(" ")), line


def test_summary_reports_degraded_flag():
    """降级轮次必须能一眼看出来 —— 排查「为什么这次走了兜底」全靠它。"""
    collector = TurnTiming()
    collector.record("llm", 12.0)
    collector.path = "handoff_lowconf"
    line = format_summary(collector, degraded=True)
    assert "degraded=1" in line
    assert "path=handoff_lowconf" in line


def test_summary_omits_zero_stages():
    """没跑过的阶段不该出现在日志里（0ms 会让人误以为跑过）。"""
    collector = TurnTiming()
    collector.record("llm", 5.0)
    line = format_summary(collector)
    assert "llm=" in line
    assert "embed=" not in line and "rerank=" not in line


def test_summary_declares_inclusive_scope():
    """日志行必须自带口径标记 `口径=total_incl`。

    这不是给读源码的人看的，是给**看日志的人**看的：没有它，运维看到
    `fusion=1098ms` 与 `vector_search=5512ms` 并列会怀疑数据坏了，
    更糟的是把各阶段相加做容量规划 —— 嵌套部分会被重复计入。
    实测真实链路里 fusion 确实可能小于 vector_search（建连/建索引在 fusion 之外），
    所以「只有 total 是超集」这条信息必须出现在日志本身。
    """
    collector = TurnTiming()
    collector.record("fusion", 1098.0)
    collector.record("vector_search", 5512.0)
    line = format_summary(collector)
    assert "口径=total_incl" in line, f"日志行缺口径标记：{line}"
    # 口径标记必须在阶段段之前，扫一眼就能看到
    assert line.index("口径=total_incl") < line.index("阶段:"), line


def test_summary_marks_no_stage_explicitly():
    """一个阶段都没跑时必须显式写 `阶段: 无`，不能留空。

    渠道未启用 / 会话已停止这类分支会在任何埋点之前返回。留空会被读成
    「阶段都很快」，而真实含义是「没走到任何有开销的环节」——
    两者的排障结论完全不同，所以不能靠「path 字段去猜」。
    """
    line = format_summary(TurnTiming())
    assert "阶段: 无" in line, f"无阶段时未显式标注：{line}"
    assert "阶段_calls" not in line, line


def test_log_summary_silent_without_turn(caplog):
    """不在一轮对话里时不得抛异常、不得打日志。"""
    with caplog.at_level(logging.INFO, logger="aics.test"):
        assert log_summary(logging.getLogger("aics.test"), None) == ""
    assert "链路耗时" not in caplog.text


def test_total_is_frozen_after_first_read():
    """total 首次读取后定格：同一轮打两次摘要必须是同一个数字，否则对不上账。"""
    with turn() as collector:
        with stage("llm"):
            pass
        first = collector.total_ms()
    time.sleep(0.02)
    assert collector.total_ms() == first, "total 不该在定格后继续增长"


# --------------------------------------------------------------------------- #
# 端到端：一轮 handle_turn 必须打出含阶段 key 的单行摘要
# --------------------------------------------------------------------------- #
def _timing_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("链路耗时")]


@pytest.fixture
def timing_capture(caplog):
    with caplog.at_level(logging.INFO, logger="aics.agent"):
        yield caplog


# 这些字段决定「走正常生成还是兜底」，而演示员工是 session 级共享的：
# 改了不还原的话，后面的用例会踩到前一个用例留下的阈值（表现为莫名其妙的
# handoff / tokens 为 None）。快照-还原比每个用例各写一遍finally 更难写错。
_EMP_CFG_FIELDS = (
    "low_confidence_policy", "confidence_threshold",
    "clarify_rounds", "auto_stop_enabled", "auto_stop_threshold",
)


@pytest.fixture
def employee_cfg(client, star):
    """给默认员工改配置，返回一个「改完自动还原」的 setter。"""
    emp_id = _default_employee(client, star)["id"]

    def _snapshot() -> dict:
        return client.get(f"/api/employees/{emp_id}", headers=star).json()

    before = {k: _snapshot()[k] for k in _EMP_CFG_FIELDS}

    def _set(**fields):
        res = client.patch(f"/api/employees/{emp_id}", headers=star, json=fields)
        assert res.status_code == 200, res.text

    yield _set

    after = _snapshot()
    restore = {k: before[k] for k in _EMP_CFG_FIELDS if after.get(k) != before[k]}
    if restore:
        res = client.patch(f"/api/employees/{emp_id}", headers=star, json=restore)
        assert res.status_code == 200, res.text


@pytest.fixture
def confident_retrieve(monkeypatch):
    """把检索桩成「高置信命中」，强制走正常生成分支。

    为什么必须桩：演示知识库是**全测试套件共享**的，其他用例（删文档、
    删知识库）跑完会把它清空。不桩的话本用例的走向取决于测试执行顺序 ——
    单独跑绿、跟着全量跑红，而且这种红是「顺序依赖」不是真bug，最难查。
    """
    import app.agent as agent_mod
    from app.rag import RetrievedChunk

    async def fake_retrieve(db, **kw):
        return [
            RetrievedChunk(
                chunk_id="chunk-timing-1", kb_id=kw.get("kb_ids", ["kb"])[0] if kw.get("kb_ids") else "kb",
                doc_id="doc-timing-1", filename="测试文档.pdf", score=0.93,
                text="户外电源保修期为 12 个月，人为损坏不在保修范围内。",
            )
        ], 5

    monkeypatch.setattr(agent_mod, "retrieve", fake_retrieve)
    return fake_retrieve


def test_handle_turn_emits_single_summary_line(
    client, star, monkeypatch, timing_capture, confident_retrieve
):
    """正常一轮：必须恰好一行摘要，且带齐关键字段与真实阶段。"""
    import app.agent as agent_mod
    from app.llm import LLMResult

    async def fake_chat(messages, **kw):
        return LLMResult(
            text="这款网桥净重 420g，支持 2.4G/5G 双频。",
            prompt_tokens=7, completion_tokens=11, endpoint="stub",
        )

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    before = len(_timing_lines(timing_capture))

    res = client.post(
        "/api/chat/message", headers={"X-Widget-Key": key},
        json={"message": "户外电源保修多久", "visitor_id": "v-timing-normal"},
    ).json()
    assert res.get("reply"), f"本轮应正常作答：{res}"

    lines = _timing_lines(timing_capture)[before:]
    assert len(lines) == 1, f"一轮应只打一行摘要，实际 {len(lines)} 行：{lines}"
    line = lines[0]

    # 定位字段 + 端到端 + 分支：grep 到这一行就能定位到「谁、慢不慢、走的哪条路」
    assert "tenant=" in line and "emp=" in line and "origin=chat" in line
    assert re.search(r"total=\d+(\.\d+)?ms", line), line
    assert "path=normal" in line, line
    assert "degraded=0" in line, line

    # 阶段：意图判定与提示词拼装是纯 CPU，但必须真的被埋到
    assert "intent=" in line, f"缺 intent 阶段：{line}"
    assert "prompt_build=" in line, f"缺 prompt_build 阶段：{line}"
    assert "llm=" in line and "llm_calls=1" in line, f"缺 llm 阶段/次数：{line}"
    assert "history=" in line, f"缺 history 阶段：{line}"
    # token 也要出现在耗时行里：耗时与花费对照着看才知道「慢是不是因为输出太长」
    assert "tokens=" in line, f"缺 tokens 字段：{line}"


def test_handle_turn_summary_covers_early_return_paths(client, star, monkeypatch, timing_capture):
    """早退分支也必须出摘要，且 path 要能区分是「转人工」还是「无意义输入」。

    这是最容易被漏掉的一类：早退分支代码最短、最容易忘记加埋点，
    而恰恰是「转人工了但没人知道卡在哪」最需要这行日志。
    """
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    seen_paths: list[str] = []

    def _grab(visitor_id: str, message: str) -> str:
        before = len(_timing_lines(timing_capture))
        client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": message, "visitor_id": visitor_id},
        )
        lines = _timing_lines(timing_capture)[before:]
        assert len(lines) == 1, f"「{message}」这轮应出一行摘要：{lines}"
        path = re.search(r"path=(\S+)", lines[0]).group(1)
        seen_paths.append(path)
        return lines[0]

    # 无意义输入：最早的 return 点之一（intent 判定之后立刻返回）
    _grab("v-timing-early-1", "在吗")
    # 强制转人工：走 intent 之后的 return 点
    _grab("v-timing-early-2", "转人工")

    assert "meaningless" in seen_paths[0], seen_paths
    assert "handoff_forced" in seen_paths[1], seen_paths
    # 早退分支也要带 total 与定位字段
    assert "tenant=" in _timing_lines(timing_capture)[-1]


def test_handle_turn_summary_reports_degraded_low_confidence(
    client, star, monkeypatch, timing_capture, employee_cfg
):
    """低置信降级轮：degraded=1 + 对应 path 必须同时出现在摘要里。

    降级原因排查靠的就是这两个字段 —— 只有耗时没有 degraded/path，
    看到「这轮 9 秒」根本不知道它是正常生成还是兜底话术。
    """
    import app.agent as agent_mod
    from app.llm import LLMResult

    employee_cfg(low_confidence_policy="handoff", confidence_threshold=95)

    async def fake_chat(messages, **kw):
        return LLMResult(text="正常回答", prompt_tokens=1, completion_tokens=1, endpoint="stub")

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    before = len(_timing_lines(timing_capture))
    client.post(
        "/api/chat/message", headers={"X-Widget-Key": key},
        json={"message": "明天天气怎么样", "visitor_id": "v-timing-lowconf"},
    )
    lines = _timing_lines(timing_capture)[before:]
    assert len(lines) == 1, lines
    assert "degraded=1" in lines[0], lines[0]
    assert "path=handoff_lowconf" in lines[0], lines[0]


def test_rag_stages_are_instrumented():
    """embed / vector_search 必须埋在 rag.retrieve 内部，不是在 agent 层包一层 retrieve。

    钉这条是为了防「有人在agent 层给 retrieve 整体计时」这种回退做法：
    那样只能看到一个 retrieve 总耗时，运维分不出慢在嵌入模型还是向量库。
    这里直接调rag.retrieve 并检查各阶段的 calls —— 真实链路下
    embed/vector_search 各调1 次，calls 能证明埋点落在了正确的调用点上
    （包在 retrieve 外面的话，这两个 key 根本不会出现）。
    """
    import asyncio

    from app import rag
    from app.database import SessionLocal

    async def _run() -> dict:
        db = SessionLocal()
        try:
            with timing.turn(prefix="tenant=t-rag") as collector:
                try:
                    await rag.retrieve(
                        db, tenant_id="t-rag", kb_ids=["kb-timing"], query="保修多久",
                    )
                except Exception:  # noqa: BLE001 - 无语料/无集合是预期内的
                    pass
                return {"ms": dict(collector.stage_ms), "calls": dict(collector.stage_calls)}
        finally:
            db.rollback()
            db.close()

    got = asyncio.run(_run())
    assert "embed" in got["calls"], f"rag.retrieve 内部缺 embed 埋点：{got}"
    assert got["calls"]["embed"] == 1, f"embed 埋点位置不对（应为 1 次）：{got}"
    assert "vector_search" in got["calls"], f"rag.retrieve 内部缺 vector_search 埋点：{got}"
    assert got["calls"]["vector_search"] >= 1, got


def test_rag_vector_search_counts_per_store_call():
    """多库模式下每次 store.search 都要单独计时（vector_search_calls = 库数）。

    合成一个 stage 会把「调了几次」这个关键信息抹掉：多库各查一次的总耗时
    与单库一次查完的耗时，数值可能一样，排障结论却完全相反。
    """
    import asyncio

    from app import rag
    from app.database import SessionLocal
    from app.vectorstore.base import VectorHit

    class _Store:
        def __init__(self):
            self.calls = 0

        def search(self, tenant_id, vec, **kw):
            self.calls += 1
            return [VectorHit(chunk_id=f"c{self.calls}", score=0.0)]

    store = _Store()

    async def _run() -> dict:
        db = SessionLocal()
        try:
            with timing.turn(prefix="tenant=t-multi") as collector:
                with mock.patch.object(rag, "build_vector_store", lambda *a, **k: store):
                    try:
                        await rag.retrieve(
                            db, tenant_id="t-multi",
                            kb_ids=["kb-a", "kb-b", "kb-c"], query="保修",
                        )
                    except Exception:  # noqa: BLE001 - 无语料是预期内的
                        pass
                return dict(collector.stage_calls)
        finally:
            db.rollback()
            db.close()

    calls = asyncio.run(_run())
    assert calls.get("vector_search") == 3, (
        f"3 个库应记 3 次向量库查询，实际 {calls.get('vector_search')} 次：{calls}"
    )


def test_llama_engine_stages(monkeypatch):
    """llama 引擎的阶段（bm25 / fusion / vector_search）必须各自计时。

    这条直接对应生产配置（RAG_ENGINE=llama），而它的检索是多路的：
    只有一个笼统的 retrieve 耗时的话，「Qdrant 慢」和「BM25 建索引慢」
    这两种完全不同的问题会被混成同一个数字。
    """
    from app.database import SessionLocal
    from app import rag_llama

    class _Retriever:
        """假检索器：在 aretrieve 里埋自己的 stage，验证融合器能把它们分开计时。"""

        def __init__(self, key: str):
            self.key = key

        async def aretrieve(self, qb):
            with timing.stage(self.key):
                await asyncio.sleep(0.01)
            return []

    async def _fake_embed(query):
        return [0.0], 3

    async def _run() -> dict[str, float]:
        monkeypatch.setattr(rag_llama, "embed_query", _fake_embed)
        monkeypatch.setattr(rag_llama, "llama_vector_store", lambda: object())
        monkeypatch.setattr(rag_llama, "AicsVectorRetriever", lambda *a, **k: _Retriever("vector_search"))
        monkeypatch.setattr(rag_llama, "_bm25_retriever", lambda *a, **k: _Retriever("bm25"))
        monkeypatch.setattr(rag_llama, "_pinned_key_chunks", lambda *a, **k: [])
        monkeypatch.setattr(rag_llama, "_scope_chunks", lambda *a, **k: [])
        monkeypatch.setattr(rag_llama, "_service_supplement", lambda *a, **k: [])
        db = SessionLocal()
        try:
            with timing.turn(prefix="tenant=llama") as collector:
                await rag_llama.retrieve(
                    db, tenant_id="t-llama", kb_ids=["kb-x"], query="保修多久",
                )
                return dict(collector.stage_ms)
        finally:
            db.rollback()
            db.close()

    captured = asyncio.run(_run())

    assert "embed" in captured, f"llama 引擎缺 embed 埋点：{captured}"
    assert captured.get("fusion", 0) > 0, f"llama 引擎缺 fusion 埋点：{captured}"
    # 两路检索器在 asyncio.gather 里并发跑完，contextvars 必须让它们的埋点
    # 都落进同一个收集器 —— 否则并发下只剩一路的数据，排查会误导。
    assert captured.get("bm25", 0) > 0, f"llama 引擎缺 bm25 埋点：{captured}"
    assert captured.get("vector_search", 0) > 0, f"llama 引擎缺 vector_search 埋点：{captured}"


def test_llama_retrievers_have_stage_instrumentation():
    """源码级钉住：llama 的两个检索器内部各自埋了自己的阶段。

    真实的 AicsVectorRetriever / JiebaBM25Retriever 要连Qdrant 与语料才跑得起来，
    端到端跑不到；但「埋点写在不在正确的函数里」这件事可以静态断言 ——
    写错位置（比如把 vector_search 埋到融合器外面）会让两路的耗时都算进 fusion，
    于是 log 里再也分不出是哪一路慢。
    """
    import inspect

    from app import rag_llama

    vec_src = inspect.getsource(rag_llama.AicsVectorRetriever._retrieve)
    assert 'timing.stage("vector_search")' in vec_src, "向量路埋点位置不对"
    bm_src = inspect.getsource(rag_llama.JiebaBM25Retriever._retrieve)
    assert 'timing.stage("bm25")' in bm_src, "BM25 路埋点位置不对"


def test_llama_fusion_is_not_a_superset_of_the_two_paths():
    """回归：`fusion` **不是** `vector_search`/`bm25` 的超集，别把它当"检索总耗时"。

    这条钉的是一个真实踩过的坑：最初以为 fusion 覆盖两路，于是在 docstring 里
    写「fusion 覆盖向量路与 BM25 路各自的耗时」。实际结构是——
    「建连探测 + BM25 建索引」在 fusion **之外**，「查询 + 检索」在 fusion **之内**。
    于是真实日志会出现 `fusion=1098ms` 而 `vector_search=5512ms`。

    后果有二，都必须防住：
    1. 有人照着错误注释做容量规划，把两路相加当成检索开销；
    2. 有人以为数据坏了，进而怀疑整个埋点不可信。

    所以：注释必须写明非超集，日志行必须自带 `口径=total_incl`。
    这里断言源码里那份口径说明还在（防止后来人又改回错误描述）。
    """
    import inspect

    import app.timing as timing_mod
    from app import rag_llama

    # fusion 的埋点必须在「建连探测 + 建索引」之后 —— 顺序反了就变成超集了
    retrieve_src = inspect.getsource(rag_llama.retrieve)
    assert retrieve_src.index('timing.stage("bm25")') < retrieve_src.index(
        'timing.stage("fusion")'
    ), "BM25 建索引的埋点应排在 fusion 之前"
    assert retrieve_src.index('timing.stage("vector_search")') < retrieve_src.index(
        'timing.stage("fusion")'
    ), "向量路建连探测的埋点应排在 fusion 之前"

    # 模块级 docstring 必须写明「只有 total 是包含式」
    doc = timing_mod.__doc__ or ""
    assert "只有 `total` 是包含式的" in doc, "口径说明丢失了"
    assert "fusion" in doc and "不是" in doc, "未说明 fusion 非超集"


# --------------------------------------------------------------------------- #
# 回归：埋点不得改变任何既有行为
# --------------------------------------------------------------------------- #
def test_timing_does_not_change_reply_or_tokens(
    client, star, monkeypatch, timing_capture, confident_retrieve
):
    """同一段桩对话，埋点前后必须给出**逐字段相同**的结果。

    这是本改动的核心回归防线。计时器一旦不小心 `return` 了什么、
    或者某次 `with` 把 `result` 的赋值吞掉，这里就会红。
    """
    import app.agent as agent_mod
    from app.llm import LLMResult

    async def fake_chat(messages, **kw):
        return LLMResult(
            text="这款网桥净重 420g，支持 2.4G/5G 双频。",
            prompt_tokens=7, completion_tokens=11, endpoint="stub",
        )

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    def ask(visitor_id: str) -> dict:
        return client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": "户外电源保修多久", "visitor_id": visitor_id},
        ).json()

    with_timing = ask("v-timing-cmp-a")

    # 把计时器彻底关掉再问一次：log_summary 拿不到 collector 就静默返回，
    # stage() 也退化成空操作 —— 剩下的就是「未埋点时的行为」。
    real_log_summary = timing.log_summary
    try:
        monkeypatch.setattr(timing, "log_summary", lambda *a, **k: "")
        without_timing = ask("v-timing-cmp-b")
    finally:
        monkeypatch.setattr(timing, "log_summary", real_log_summary)

    for field in ("reply", "handoff", "handoff_reason", "status",
                  "usage_total_tokens", "degraded", "degrade_reason"):
        assert with_timing.get(field) == without_timing.get(field), (
            f"埋点改变了 {field}：{with_timing.get(field)!r} != {without_timing.get(field)!r}"
        )


def test_timing_does_not_change_handoff_decision(
    client, star, monkeypatch, timing_capture, employee_cfg
):
    """转人工判定不得受埋点影响（handoff / status / handoff_reason 三者全等）。"""
    employee_cfg(low_confidence_policy="handoff", confidence_threshold=95)
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    def ask(visitor_id: str) -> dict:
        return client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": "转人工", "visitor_id": visitor_id},
        ).json()

    with_timing = ask("v-timing-ho-a")
    real_log_summary = timing.log_summary
    try:
        monkeypatch.setattr(timing, "log_summary", lambda *a, **k: "")
        without_timing = ask("v-timing-ho-b")
    finally:
        monkeypatch.setattr(timing, "log_summary", real_log_summary)

    for field in ("handoff", "handoff_reason", "status", "reply"):
        assert with_timing.get(field) == without_timing.get(field), (
            f"埋点改变了 {field}：{with_timing.get(field)!r} != {without_timing.get(field)!r}"
        )


def test_timing_does_not_change_token_accounting(
    client, star, monkeypatch, timing_capture, employee_cfg, confident_retrieve
):
    """计费 token 数不得受埋点影响 —— 少算就是真金白银的账错了。"""
    import app.agent as agent_mod
    from app.llm import LLMResult

    # 阈值放到最低，保证这一轮真的走「正常生成」（tokens 只在正常生成/兜底作答时计入）
    employee_cfg(confidence_threshold=1, low_confidence_policy="clarify")

    async def fake_chat(messages, **kw):
        return LLMResult(
            text="答复", prompt_tokens=13, completion_tokens=17, endpoint="stub",
        )

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])

    with_timing = client.post(
        "/api/chat/message", headers={"X-Widget-Key": key},
        json={"message": "户外电源保修多久", "visitor_id": "v-timing-tok-a"},
    ).json()
    real_log_summary = timing.log_summary
    try:
        monkeypatch.setattr(timing, "log_summary", lambda *a, **k: "")
        without_timing = client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": "户外电源保修多久", "visitor_id": "v-timing-tok-b"},
        ).json()
    finally:
        monkeypatch.setattr(timing, "log_summary", real_log_summary)

    assert with_timing.get("usage_total_tokens") == without_timing.get("usage_total_tokens"), (
        f"埋点改变了 token 计数：{with_timing.get('usage_total_tokens')} "
        f"!= {without_timing.get('usage_total_tokens')}"
    )
    assert (with_timing.get("usage_total_tokens") or 0) > 0, \
        "前提：这一轮应有 token 消耗，否则断言无意义"


def test_every_return_point_sets_a_path():
    """回归：handle_turn 里每个 `return result` 前都必须登记 path。

    漏一个就等于该分支永远打不出 `path=`（会打成 unknown），
    排障时「这轮到底走没走兜底」就答不上来了。这里做源码级静态检查，
    比逐个分支跑一遍更防回归 —— 新加的 return 点会立刻被抓到。
    """
    import inspect

    import app.agent as agent_mod

    src = inspect.getsource(agent_mod.handle_turn.__wrapped__)
    lines = src.splitlines()
    missing: list[int] = []
    for idx, line in enumerate(lines):
        if line.strip() != "return result":
            continue
        window = "\n".join(lines[max(0, idx - 4):idx])
        if "result.path = " not in window:
            missing.append(idx + 1)
    assert not missing, f"这些 return 点没登记 path（行号相对函数体）：{missing}"


def test_all_declared_paths_are_used():
    """反向检查：登记在案的 path 常量必须都被用到（防拼错或改完忘改另一处）。"""
    import app.agent as agent_mod
    import inspect

    src = inspect.getsource(agent_mod)
    declared = re.findall(r"^(_PATH_[A-Z_]+) = ", src, flags=re.M)
    assert declared, "应至少声明一个 _PATH_* 常量"
    for name in declared:
        assert src.count(name) >= 2, f"{name} 声明后未被使用"


def test_stage_keys_are_unique():
    """STAGE_KEYS 不得有重复 —— 重复会让日志里同名阶段出现两次，解析错位。"""
    assert len(timing.STAGE_KEYS) == len(set(timing.STAGE_KEYS))


def test_stage_keys_cover_declared_stages():
    """agent / rag / rag_llama 里实际用到的 stage key 都必须登记在 STAGE_KEYS。

    未登记的 key 会被兼容逻辑兜着打出来（不至于丢数据），但顺序不稳定，
    所以这里钉住：新增埋点必须同步登记。
    """
    import inspect

    import app.agent as agent_mod
    import app.rag as rag_mod
    import app.rag_llama as rag_llama_mod

    used: set[str] = set()
    for mod in (agent_mod, rag_mod, rag_llama_mod):
        used |= set(re.findall(r'stage\(\s*"([a-z_]+)"', inspect.getsource(mod)))
    missing = used - set(timing.STAGE_KEYS)
    assert not missing, f"这些阶段没登记到 STAGE_KEYS：{sorted(missing)}"