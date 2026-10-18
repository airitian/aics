"""低置信兜底：礼貌式拒答必须被识别为「未有效回答」，否则兜底策略永远不触发。

背景（本文件针对的真实事故）：
概览/上下文兜底是「模型确认答不出才往下走」的中间层，靠 `NO_ANSWER` 哨兵把关。
但哨兵只是**建议**：线上模型并不总是老实地回哨兵，而是照着文档概览硬凑一句
客套拒答（"这个我确实查不了哈，我这边主要负责产品咨询和选购这块～"）。
旧判定只排除哨兵，于是把这种回复当成有效回答放行 → `no_answer_streak` /
`clarify_count` 被清零 → 追问与转人工分支永远到不了 → 会话卡在概览兜底出不来。

这里钉住三件事：
1. `is_refusal` 能认出真实的礼貌拒答；
2. **不误杀**正常产品答复（"人为损坏不在保修范围内"是有效答案，不是拒答）；
3. 计数无条件累加（`auto_stop_enabled=False` 也不能让计数停摆）。
"""
from __future__ import annotations

import pytest

from app.agent import _is_real_answer, escalate_no_answer, is_refusal
from app.models import AiEmployee
from app.models import Session as ChatSession

from tests.test_flows import _default_employee, _widget_key


# --------------------------------------------------------------------------- #
# 拒答识别：正例（线上实测原文）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    # 线上复现输出原文 —— 改判定逻辑时这一条必须仍然通过
    "这个我确实查不了哈，我这边主要负责产品咨询和选购这块～",
    "抱歉，我不太清楚您具体想了解什么呢。",
    "这个我们这边没有相关资料呢，建议您咨询人工客服。",
    "不好意思，我不太方便回答这个问题。",
    "帮不上您这个忙。",
    "这个不在我服务范围内。",
    "我无法回答这个问题。",
    "我没查到相关内容",
    "不太清楚",
])
def test_refusal_detected(text):
    assert is_refusal(text) is True, f"礼貌拒答未被识别：{text!r}"


# --------------------------------------------------------------------------- #
# 反向：正常产品答复绝不能被误判成拒答（这是本判定的最大风险）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "这款网桥净重420g，尺寸 120x80x25mm，工作温度 -20~60℃。",
    # 这一条曾经被 `不(在|属于).{0,4}范围` 误杀：「损坏不在保修范围内」
    # 是在**陈述保修条款**，是有效的产品答复，不是模型自述无能。
    "保修期为12个月，人为损坏不在保修范围内。",
    "您好这款不支持 5G 频段，只支持 2.4G。",
    "支持 2.4G 和 5G 双频，5GHz 频段支持 802.11ax。",
    "I do not have information about a product called the WB730.",
    "这款产品目前不支持定制颜色。",
    "我们这款不在 5G 频段范围内，仅支持 2.4G。",
])
def test_real_answer_not_flagged_as_refusal(text):
    assert is_refusal(text) is False, f"正常答复被误判为拒答：{text!r}"
    assert _is_real_answer(text) is True


# --------------------------------------------------------------------------- #
# 放行闸门：哨兵与拒答都不算有效回答
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", "   ", "NO_ANSWER", "这个我确实查不了哈"])
def test_gate_rejects_non_answers(text):
    assert _is_real_answer(text) is False, f"不该放行：{text!r}"


# --------------------------------------------------------------------------- #
# 计数：开关关掉也必须累加
# --------------------------------------------------------------------------- #
def test_streak_counts_even_when_auto_stop_disabled():
    """回归：`auto_stop_enabled=False` 时计数也必须累加。

    之前本函数在开关为假时既不计数也不转人工 —— 运维关掉开关后，
    计数停摆，再打开时面对的是一个「从没发生过」的会话，阈值判断完全失真。
    """
    session = ChatSession(id="s1", tenant_id="t1", employee_id="e1", visitor_id="v1")
    emp = AiEmployee(
        id="e1", tenant_id="t1", name="测试员工",
        auto_stop_enabled=False, auto_stop_threshold=3,
    )
    assert escalate_no_answer(session, emp) is False, "开关关着不该转人工"
    assert session.no_answer_streak == 1, "但计数必须累加"
    assert escalate_no_answer(session, emp) is False
    assert session.no_answer_streak == 2
    assert escalate_no_answer(session, emp) is False, "开关关着到阈值也不转人工"
    assert session.no_answer_streak == 3, "计数仍应累加到 3"


def test_streak_triggers_handoff_when_enabled():
    """开关打开时，阈值一到就转人工。"""
    session = ChatSession(id="s2", tenant_id="t1", employee_id="e1", visitor_id="v2")
    emp = AiEmployee(
        id="e1", tenant_id="t1", name="测试员工",
        auto_stop_enabled=True, auto_stop_threshold=3,
    )
    assert escalate_no_answer(session, emp) is False
    assert escalate_no_answer(session, emp) is False
    assert escalate_no_answer(session, emp) is True, "第 3 次应触发转人工"
    assert session.handoff_reason, "转人工必须带原因"
    assert "3" in session.handoff_reason


# --------------------------------------------------------------------------- #
# 端到端：连续无解必须真的转人工（而不是永远卡在概览兜底）
# --------------------------------------------------------------------------- #
def test_consecutive_unanswerable_hands_off(client, star, monkeypatch):
    """模型每轮都礼貌拒答 → 追问用尽后必须转人工，且不再卡在概览兜底。

    这里刻意让桩模型**回礼貌拒答而不是 NO_ANSWER 哨兵** ——
    哨兵路径本来就能走到追问/转人工，测它等于没测到本 bug。
    """
    import app.agent as agent_mod
    from app.llm import LLMResult

    emp = _default_employee(client, star)
    client.patch(
        f"/api/employees/{emp['id']}", headers=star,
        json={
            "low_confidence_policy": "clarify", "clarify_rounds": 1,
            "confidence_threshold": 50, "auto_stop_threshold": 3,
            "auto_stop_enabled": False,
        },
    )
    emp = _default_employee(client, star)
    assert emp["low_confidence_policy"] == "clarify"
    assert emp["clarify_rounds"] == 1

    refusal = "这个我确实查不了哈，我这边主要负责产品咨询和选购这块～"

    async def fake_chat(messages, **kw):
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "【兜底说明】" in text or "【对话上下文兜底】" in text:
            # 复现线上真实行为：照着概览硬凑一句客套拒答，不回哨兵
            return LLMResult(text=refusal, prompt_tokens=1, completion_tokens=1, endpoint="stub")
        return LLMResult(
            text="这款网桥净重 420g，支持 2.4G/5G 双频。",
            prompt_tokens=1, completion_tokens=1, endpoint="stub",
        )

    monkeypatch.setattr(agent_mod, "chat", fake_chat)

    key = _widget_key(client, star, emp["id"])
    replies = []
    for q in ("明天天气怎么样", "今天限号吗", "你们老板叫什么"):
        replies.append(client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": q, "visitor_id": "v-lowconf-regression"},
        ).json())

    # 关键回归断言：任何一轮都不允许出现「假装答了」的概览放行
    for i, r in enumerate(replies, 1):
        reason = r.get("degrade_reason") or ""
        assert "概览兜底作答" not in reason, (
            f"第 {i} 轮仍把礼貌拒答当成有效回答放行 —— 本 bug 未修复：{reason}"
        )

    # 追问用尽后必须真的转人工
    assert replies[-1]["handoff"] is True, f"连续无解后未转人工：{replies[-1]}"
    assert replies[-1]["status"] == "queued"
    assert replies[-1]["handoff_reason"], "转人工必须带原因"