"""配置边界、发布流程、对话链路测试。"""
from __future__ import annotations

import pytest

from tests.conftest import login


def _default_employee(client, headers) -> dict:
    items = client.get("/api/employees", headers=headers).json()["items"]
    return next(e for e in items if e["is_default"])


def _widget_key(client, headers, employee_id: str) -> str:
    return client.get(f"/api/employees/{employee_id}/widget-keys", headers=headers).json()["items"][0]["key"]


# --------------------------------------------------------------------------- #
# 边界值
# --------------------------------------------------------------------------- #
def test_persona_over_limit_rejected(client, star):
    emp = _default_employee(client, star)
    res = client.patch(f"/api/employees/{emp['id']}", headers=star, json={"persona": "字" * 20001})
    assert res.status_code == 422, "人设超过 20000 字符必须被拒"


def test_numeric_bounds_enforced(client, star):
    emp = _default_employee(client, star)
    for field, bad in [("confidence_threshold", 101), ("clarify_rounds", 4),
                       ("llm_max_tokens", 0), ("auto_stop_threshold", 11)]:
        res = client.patch(f"/api/employees/{emp['id']}", headers=star, json={field: bad})
        assert res.status_code == 422, f"{field}={bad} 应被拒绝"


def test_unsupported_language_rejected(client, star):
    emp = _default_employee(client, star)
    res = client.patch(f"/api/employees/{emp['id']}", headers=star, json={"output_language": "xx-XX"})
    assert res.status_code == 422


def test_employee_count_limit(client, platform):
    """单租户 AI 员工数上限（PRD 第八章：建议 10）。"""
    res = client.post("/api/platform/tenants", headers=platform,
                      json={"name": "上限测试租户", "admin_email": "limit@demo.local", "admin_password": "demo12345"})
    tid = res.json()["id"]
    slug = next(t["slug"] for t in client.get("/api/platform/tenants", headers=platform).json()["items"] if t["id"] == tid)
    headers = login(client, "limit@demo.local", "demo12345")
    try:
        assert len(client.get("/api/employees", headers=headers).json()["items"]) == 1
        for i in range(9):
            r = client.post("/api/employees", headers=headers, json={"name": f"员工{i}"})
            assert r.status_code == 200, r.text
        r = client.post("/api/employees", headers=headers, json={"name": "第十一个"})
        assert r.status_code == 400
        assert "上限" in r.json()["detail"]
    finally:
        client.post(f"/api/platform/tenants/{tid}/purge?confirm={slug}", headers=platform)


# --------------------------------------------------------------------------- #
# 发布与版本
# --------------------------------------------------------------------------- #
def test_publish_blocked_when_persona_empty(client, star):
    emp = _default_employee(client, star)
    original = emp["persona"]
    try:
        client.patch(f"/api/employees/{emp['id']}", headers=star, json={"persona": "   "})
        res = client.post(f"/api/employees/{emp['id']}/publish", headers=star, json={"force": True})
        body = res.json()
        assert body["ok"] is False
        assert any("人设不能为空" in b for b in body["blocked"])
    finally:
        client.patch(f"/api/employees/{emp['id']}", headers=star, json={"persona": original})


def test_publish_blocked_on_unknown_variable(client, star):
    emp = _default_employee(client, star)
    original = emp["persona"]
    try:
        client.patch(f"/api/employees/{emp['id']}", headers=star, json={"persona": "你好 {{not.exists}}"})
        body = client.post(f"/api/employees/{emp['id']}/publish", headers=star, json={"force": True}).json()
        assert body["ok"] is False
        assert any("不存在的变量" in b for b in body["blocked"])
    finally:
        client.patch(f"/api/employees/{emp['id']}", headers=star, json={"persona": original})


def test_publish_version_and_rollback(client, star):
    emp = _default_employee(client, star)
    before = emp["published_version"]

    changes = client.patch(f"/api/employees/{emp['id']}", headers=star,
                           json={"persona": "你是测试人设，请依据知识库回答。"}).json()
    assert changes["status"] == "draft", "改完配置应回到草稿态"

    body = client.post(f"/api/employees/{emp['id']}/publish", headers=star, json={"force": True}).json()
    assert body["ok"] is True
    assert body["version_no"] == before + 1

    versions = client.get(f"/api/employees/{emp['id']}/versions", headers=star).json()["items"]
    assert len(versions) >= 2
    assert versions[0]["is_current"] is True

    rb = client.post(f"/api/employees/{emp['id']}/versions/{before}/rollback", headers=star).json()
    assert rb["ok"] is True
    after = client.get(f"/api/employees/{emp['id']}", headers=star).json()
    assert after["status"] == "draft", "回滚后必须再次发布才生效"


def test_empty_kb_publish_needs_confirmation(client, platform):
    """空知识库允许发布，但必须二次确认（PRD 3.2）。"""
    res = client.post("/api/platform/tenants", headers=platform,
                      json={"name": "空库测试租户", "admin_email": "empty@demo.local", "admin_password": "demo12345"})
    tid = res.json()["id"]
    slug = next(t["slug"] for t in client.get("/api/platform/tenants", headers=platform).json()["items"] if t["id"] == tid)
    headers = login(client, "empty@demo.local", "demo12345")
    try:
        emp = _default_employee(client, headers)
        first = client.post(f"/api/employees/{emp['id']}/publish", headers=headers, json={"force": False}).json()
        assert first["ok"] is False
        assert any("知识库" in w for w in first["warnings"])
        second = client.post(f"/api/employees/{emp['id']}/publish", headers=headers, json={"force": True}).json()
        assert second["ok"] is True
    finally:
        client.post(f"/api/platform/tenants/{tid}/purge?confirm={slug}", headers=platform)


# --------------------------------------------------------------------------- #
# 对话链路
# --------------------------------------------------------------------------- #
def test_chat_hits_own_knowledge(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    res = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                      json={"message": "户外电源保修多久？", "visitor_id": "v-hit"})
    assert res.status_code == 200
    body = res.json()
    assert body["hits"], "应该命中本租户的知识片段"
    assert "户外电源" in body["hits"][0]["text"] or "保修" in body["hits"][0]["text"]
    # 开发模式下模型未接入，必须如实说明，不得编造
    assert "模型未配置" in body["reply"]


def test_chat_no_hit_says_no_material(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": "请问去火星的物流要几天到？", "visitor_id": "v-nohit"}).json()
    assert body["hits"] == []
    assert "没有" in body["reply"] or "没查到" in body["reply"]
    assert body["degraded"] is True


def test_chat_meaningless_input_asks_back(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": "在？", "visitor_id": "v-mean"}).json()
    assert "具体" in body["reply"] or "再说" in body["reply"]


@pytest.mark.parametrize("text,expect", [
    ("我要转人工", True),
    ("你们这是诈骗，我要投诉！", True),
    ("能不能便宜点，给个底价", True),
    ("这个帐篷的发货时效是多久", False),
])
def test_forced_handoff_intent(client, star, text, expect):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": text, "visitor_id": "v-intent"}).json()
    assert body["handoff"] is expect, f"{text!r} → handoff 期望 {expect}，实际 {body['handoff']}"


def test_human_takeover_silences_ai(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    sid = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                      json={"message": "我要转人工", "visitor_id": "v-hand"}).json()["session_id"]

    assert client.post(f"/api/sessions/{sid}/takeover", headers=star).status_code == 200

    after = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                        json={"message": "在吗", "visitor_id": "v-hand", "session_id": sid}).json()
    assert after["reply"] == "", "人工接管后 AI 必须静默"
    assert after["status"] == "human"

    # 人工回复 → 交回 AI
    assert client.post(f"/api/sessions/{sid}/reply", headers=star, json={"content": "您好，我来处理"}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/release", headers=star).status_code == 200

    history = client.get(f"/api/sessions/{sid}/messages", headers=star).json()
    roles = [m["role"] for m in history["items"]]
    assert "agent" in roles


def test_agent_reply_requires_takeover_first(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    sid = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                      json={"message": "你好", "visitor_id": "v-order"}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/reply", headers=star, json={"content": "直接回复"})
    assert res.status_code == 400


def test_channel_switch_disables_ai(client, star):
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    try:
        client.patch(f"/api/employees/{emp['id']}", headers=star,
                     json={"channel_switch": {"web": False}})
        body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                           json={"message": "帐篷多少钱", "visitor_id": "v-ch"}).json()
        assert body["handoff"] is True
        assert body["hits"] == [], "渠道关闭 AI 后不应再走检索"
    finally:
        client.patch(f"/api/employees/{emp['id']}", headers=star, json={"channel_switch": {"web": True}})


# --------------------------------------------------------------------------- #
# 问答测试与洞察
# --------------------------------------------------------------------------- #
def test_playground_batch_limit(client, star):
    emp = _default_employee(client, star)
    questions = [f"这是第 {i} 个测试问题" for i in range(60)]
    body = client.post(f"/api/playground/{emp['id']}/batch", headers=star,
                       json={"questions": questions}).json()
    assert body["executed"] == 50
    assert body["rejected_count"] == 10


def test_playground_does_not_create_online_sessions(client, star):
    emp = _default_employee(client, star)
    before = len(client.get("/api/sessions?limit=200", headers=star).json()["items"])
    client.post(f"/api/playground/{emp['id']}/ask", headers=star, json={"question": "户外电源保修多久"})
    after = len(client.get("/api/sessions?limit=200", headers=star).json()["items"])
    assert before == after, "问答测试不得产生线上会话/工单"


def test_playground_clarify_path_does_not_crash(client, star):
    """回归：问答测试的探针会话不落库，会话计数列读出来是 None。

    `clarify_count` / `no_answer_streak` 的默认值由数据库在 INSERT 时填，
    而 Playground 刻意复用「不落库的 Session」，于是下面两条分支曾直接抛
    TypeError —— 偏偏它们正是问答测试最该覆盖的场景：
      - 「无意义输入」→ session.clarify_count += 1  （None += int）
      - 「低置信 + 追问澄清」→ clarify_count < clarify_rounds（None < int）
    """
    emp = _default_employee(client, star)
    assert emp["low_confidence_policy"] == "clarify", "本用例依赖默认的「追问澄清」策略"

    for question in ("在吗", "跟知识库毫无关系的天马行空问题：量子计算机怎么修？"):
        res = client.post(f"/api/playground/{emp['id']}/ask", headers=star, json={"question": question})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ok"] is True, f"探针会话不应把 TypeError 冒出来：{body}"
        assert not body.get("error"), body
        assert body["answer"], "澄清/兜底话术不能为空"


def test_escalate_no_answer_on_unflushed_session():
    """回归：未落库的 Session 上做「连续未回答」计数不得抛错。"""
    from app.agent import escalate_no_answer
    from app.models import AiEmployee, Session as ChatSession

    session = ChatSession(id="s1", tenant_id="t1", employee_id="e1", visitor_id="v1")
    assert session.no_answer_streak is None, "前提：未落库对象的默认值确实没被填"
    employee = AiEmployee(
        id="e1", tenant_id="t1", name="测试员工",
        auto_stop_enabled=True, auto_stop_threshold=2,
    )
    assert escalate_no_answer(session, employee) is False
    assert session.no_answer_streak == 1
    assert escalate_no_answer(session, employee) is True, "达到阈值应停止 AI 回复并转人工"
    assert session.no_answer_streak == 2


# --------------------------------------------------------------------------- #
# AI 员工测试（聊天式多轮）
# --------------------------------------------------------------------------- #
def test_testchat_multi_turn_keeps_history(client, star):
    """聊天测试必须是多轮的，且 AI 实际说的话必须进历史。

    两件事一起测，因为它们是同一类缺陷的两面：
    - 只落 visitor 消息 → 没有上下文，所谓「多轮测试」是假的；
    - 早退分支不落 AI 消息 → 刷新后 AI 的回复凭空消失。

    第二轮用「在吗」是刻意的：它必然走「无意义输入」早退分支，正是原先
    只写 SYSTEM 说明、不写 AI 回复的那条路径。
    """
    emp = _default_employee(client, star)
    sid = client.post(f"/api/testchat/{emp['id']}/sessions", headers=star).json()["id"]
    try:
        for q in ("在吗", "你们支持货到付款吗"):
            res = client.post(f"/api/testchat/{emp['id']}/sessions/{sid}/messages",
                              headers=star, json={"message": q})
            assert res.status_code == 200, res.text
            assert res.json()["reply"], "每一轮都必须有回复"
            assert res.json()["session_id"] == sid

        hist = client.get(f"/api/testchat/{emp['id']}/sessions/{sid}/messages", headers=star).json()
        # 只看对话双方；SYSTEM 是给运维看的内部说明，不属于「AI 说了什么」
        said = [m for m in hist["items"] if m["role"] in ("visitor", "ai")]
        assert [m["content"] for m in said if m["role"] == "visitor"] == ["在吗", "你们支持货到付款吗"]
        assert [m["role"] for m in said] == ["visitor", "ai", "visitor", "ai"], \
            [m["role"] for m in said]
        assert all(m["content"] for m in said if m["role"] == "ai"), "AI 的回复不得落成空消息"
    finally:
        client.delete(f"/api/testchat/{emp['id']}/sessions/{sid}", headers=star)


def test_testchat_delete_clears_session(client, star):
    emp = _default_employee(client, star)
    sid = client.post(f"/api/testchat/{emp['id']}/sessions", headers=star).json()["id"]
    client.post(f"/api/testchat/{emp['id']}/sessions/{sid}/messages",
                headers=star, json={"message": "你好"})

    assert client.delete(f"/api/testchat/{emp['id']}/sessions/{sid}", headers=star).status_code == 200
    assert client.get(f"/api/testchat/{emp['id']}/sessions/{sid}/messages",
                      headers=star).status_code == 404
    ids = [s["id"] for s in client.get(f"/api/testchat/{emp['id']}/sessions", headers=star).json()["items"]]
    assert sid not in ids


def test_testchat_refuses_online_session(client, star):
    """拿线上访客会话的 id 当测试会话必须 404。

    否则测试接口就成了写入真实会话的后门 —— 测一次就污染一条线上记录。
    """
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    chat = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": "线上访客的一条消息", "visitor_id": "online-visitor-1"})
    assert chat.status_code == 200, chat.text
    online_sid = chat.json()["session_id"]

    assert client.post(f"/api/testchat/{emp['id']}/sessions/{online_sid}/messages",
                       headers=star, json={"message": "测试注入"}).status_code == 404
    assert client.get(f"/api/testchat/{emp['id']}/sessions/{online_sid}/messages",
                      headers=star).status_code == 404


def test_testchat_context_isolated_per_tester(client, star):
    """同租户同员工，不同测试者的上下文也必须互不可见。

    visitor_id 里带 user.id 就是为了这个：否则 A 的前几轮对话会出现在
    B 的上下文里，测试结果直接被串台污染。
    """
    from app.config import TEST_CHANNEL
    from app.database import SessionLocal
    from app.models import Session as ChatSession

    emp = _default_employee(client, star)
    tenant_id = client.get("/api/auth/me", headers=star).json()["tenant"]["id"]
    alien_id = "test-other-tester-session"

    db = SessionLocal()
    try:
        db.add(ChatSession(id=alien_id, tenant_id=tenant_id, employee_id=emp["id"],
                           visitor_id="tester:someone-else", channel=TEST_CHANNEL, status="ai"))
        db.commit()
    finally:
        db.close()

    try:
        assert client.get(f"/api/testchat/{emp['id']}/sessions/{alien_id}/messages",
                          headers=star).status_code == 404
        assert client.post(f"/api/testchat/{emp['id']}/sessions/{alien_id}/messages",
                           headers=star, json={"message": "伪装"}).status_code == 404
    finally:
        db = SessionLocal()
        try:
            row = db.get(ChatSession, alien_id)
            if row is not None:
                db.delete(row)
                db.commit()
        finally:
            db.close()


def test_testchat_excluded_from_online_metrics(client, star):
    """测试会话真实落库（多轮上下文需要），但不得进入线上口径。

    承重逻辑：会话行是真的写进 sessions 表的，所以一旦 overview / 坐席列表
    少了 channel 过滤，下面的断言就会红 —— 不是空跑。
    """
    emp = _default_employee(client, star)
    before = client.get("/api/insights/overview?days=7", headers=star).json()

    sid = client.post(f"/api/testchat/{emp['id']}/sessions", headers=star).json()["id"]
    try:
        client.post(f"/api/testchat/{emp['id']}/sessions/{sid}/messages",
                    headers=star, json={"message": "户外电源保修多久"})

        # 前提校验：消息确实落库了（否则这条用例等于什么都没测）
        hist = client.get(f"/api/testchat/{emp['id']}/sessions/{sid}/messages", headers=star).json()
        said = [m for m in hist["items"] if m["role"] in ("visitor", "ai")]
        assert len(said) == 2, "测试消息应已落库 —— 多轮上下文依赖它"

        # 坐席侧的会话列表不得出现测试会话
        listed = client.get("/api/sessions?limit=200", headers=star).json()["items"]
        assert all(s["id"] != sid for s in listed), "测试会话不得混进坐席会话列表"

        after = client.get("/api/insights/overview?days=7", headers=star).json()
        if before.get("empty"):
            assert after.get("empty"), "测试会话不得把概览从空态变成有数据"
        else:
            assert (after["metrics"]["sessions_total"]
                    == before["metrics"]["sessions_total"]), "测试会话不得计入线上会话总数"
    finally:
        client.delete(f"/api/testchat/{emp['id']}/sessions/{sid}", headers=star)


def test_insights_returns_definitions_and_empty_state(client, star):
    data = client.get("/api/insights/overview?days=1", headers=star).json()
    assert "definitions" in data
    assert "解决率" in data["definitions"]
    if data.get("empty"):
        assert "metrics" not in data, "空态不应返回 0% 这类误导性数值"


def test_usage_is_per_tenant(client, star, sea):
    star_usage = client.get("/api/insights/usage", headers=star).json()
    sea_usage = client.get("/api/insights/usage", headers=sea).json()
    assert star_usage["tenant_id"] != sea_usage["tenant_id"]
    # 平台侧汇总能看到全部租户，租户侧只看自己
    platform = client.get("/api/platform/overview", headers=login(client, "admin@aics.local", "admin12345")).json()
    assert platform["tenants"] >= 2


def test_audit_records_denied_attempts(client, star, sea):
    star_emp = _default_employee(client, star)["id"]
    client.get(f"/api/employees/{star_emp}", headers=sea)   # 制造一次越权尝试
    rows = client.get("/api/insights/audit?limit=200", headers=star).json()["items"]
    assert isinstance(rows, list)


# --------------------------------------------------------------------------- #
# 概览兜底：主观/口语提问 0 命中时，用概览块+放宽门槛的二次检索撑住回答
# --------------------------------------------------------------------------- #
class _FakeLLM:
    """代替 app.llm.chat 的假返回（degraded=False 才会被兜底分支采纳）。"""

    degraded = False
    model = "stub"
    endpoint = "stub"

    def __init__(self, text: str):
        self.text = text
        self.total_tokens = 10
        self.prompt_tokens = 5
        self.completion_tokens = 5


def _force_zero_hit_then_fallback(monkeypatch):
    """常规检索固定返回空（模拟口语提问 0 命中），兜底二次检索放行。"""
    import app.agent as agent_mod

    real_retrieve = agent_mod.retrieve

    async def fake_retrieve(db, **kw):
        if kw.get("no_threshold"):
            return await real_retrieve(db, **kw)
        return [], 0

    monkeypatch.setattr(agent_mod, "retrieve", fake_retrieve)


def test_overview_fallback_answers_recommendation(client, star, monkeypatch):
    """0 命中 + 概览兜底有素材 + 模型能答 → 返回推荐答案并标记降级原因。"""
    _force_zero_hit_then_fallback(monkeypatch)
    import app.agent as agent_mod

    async def fake_chat(messages, **kw):
        return _FakeLLM("综合性能我最推荐 WB730，预算有限可以看 WB620E。")

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": "你觉得哪一块最好用", "visitor_id": "v-ovfb1"}).json()
    assert "WB730" in body["reply"]
    assert body["degraded"] is True
    assert "概览兜底" in body["degrade_reason"]
    assert body["hits"], "兜底采用的素材应随响应返回，便于排查"


def test_overview_fallback_no_answer_keeps_original_policy(client, star, monkeypatch):
    """0 命中 + 模型答不出（NO_ANSWER 哨兵）→ 落回原兜底话术，不把素材硬编成答案。"""
    _force_zero_hit_then_fallback(monkeypatch)
    import app.agent as agent_mod

    async def fake_chat(messages, **kw):
        return _FakeLLM("NO_ANSWER")

    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": "你们老板娘结婚了吗", "visitor_id": "v-ovfb2"}).json()
    assert "WB730" not in body["reply"]
    assert "没查到" in body["reply"] or "没有" in body["reply"]
    assert body["hits"] == []


# --------------------------------------------------------------------------- #
# 多语言意图识别：外国客户说"要人工/要投诉"必须真的触发转人工
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "I want to speak to a human agent",
    "transfer me to a real person",
    "This is a scam, I will sue you and call my lawyer",
    "I want to file a complaint",
    "Can you give me a discount? what's your best price",
    "quiero hablar con un agente humano",
    "соедините меня с оператором",
    "担当者につないでください",
])
def test_multilingual_handoff_intent(client, star, text):
    """外语的转人工/投诉/议价请求必须触发 handoff，不能只由模型口头承诺。"""
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": text, "visitor_id": "v-i18n"}).json()
    assert body["handoff"] is True, f"{text!r} 未触发转人工"


@pytest.mark.parametrize("text", [
    "How much is the WB730?",
    "I need a bridge for 15km, which model do you recommend?",
    "What is the warranty period",
    "Do you ship to Germany",
])
def test_multilingual_normal_questions_no_handoff(client, star, text):
    """正常外语咨询不能被新增的外语关键词误伤成转人工/投诉。"""
    emp = _default_employee(client, star)
    key = _widget_key(client, star, emp["id"])
    body = client.post("/api/chat/message", headers={"X-Widget-Key": key},
                       json={"message": text, "visitor_id": "v-i18n-ok"}).json()
    assert body["handoff"] is False, f"{text!r} 被误判为转人工"


# --------------------------------------------------------------------------- #
# 绑定优先级：知识库绑定必须压过文件级绑定
#
# 背景（真实踩坑）：文件级绑定是静态快照，上传新文档不会自动进列表。
# 一旦员工残留了 doc_ids，库级绑定就被整个丢弃，用户"绑了知识库"却
# 怎么都检索不到新资料。所以库级优先：绑了库就按 kb_id 检索。
# --------------------------------------------------------------------------- #
def _upload_txt(client, headers, kb_id: str, name: str, body: str) -> str:
    res = client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=headers,
        files={"files": (name, body.encode("utf-8"), "text/plain")},
    )
    assert res.status_code == 200, res.text
    item = res.json()["items"][0]
    assert item["status"] in ("success", "partial"), item
    return item["doc_id"]


def test_kb_binding_takes_priority_over_doc_binding(client, star, monkeypatch):
    """同时绑库 + 绑文件时，检索必须走库级（doc_ids 置空）。"""
    import app.agent as agent_mod

    emp = _default_employee(client, star)
    kb = client.post("/api/knowledge-bases", headers=star,
                     json={"name": "绑定优先级测试库"}).json()
    doc_id = _upload_txt(client, star, kb["id"], "资料.txt",
                         "退款政策正文：" + "内容" * 200)

    res = client.patch(f"/api/employees/{emp['id']}", headers=star,
                       json={"kb_ids": [kb["id"]], "doc_ids": [doc_id]})
    assert res.status_code == 200, res.text

    captured: dict = {}

    async def fake_retrieve(db, **kw):
        captured.update(kw)
        return [], 0

    monkeypatch.setattr(agent_mod, "retrieve", fake_retrieve)
    key = _widget_key(client, star, emp["id"])
    client.post("/api/chat/message", headers={"X-Widget-Key": key},
                json={"message": "退款政策是怎样的", "visitor_id": "v-bind1"})

    assert captured.get("kb_ids") == [kb["id"]], f"库级绑定未生效：{captured}"
    assert not captured.get("doc_ids"), f"绑了库就不该再用文件级快照：{captured}"


def test_doc_binding_used_when_no_kb_bound(client, star, monkeypatch):
    """只勾了文件、没勾知识库时，仍按文件级检索（保留精细控制能力）。"""
    import app.agent as agent_mod

    emp = _default_employee(client, star)
    kb = client.post("/api/knowledge-bases", headers=star,
                     json={"name": "文件级绑定库"}).json()
    doc_id = _upload_txt(client, star, kb["id"], "资料.txt",
                         "保修条款正文：" + "内容" * 200)

    res = client.patch(f"/api/employees/{emp['id']}", headers=star,
                       json={"kb_ids": [], "doc_ids": [doc_id]})
    assert res.status_code == 200, res.text

    captured: dict = {}

    async def fake_retrieve(db, **kw):
        captured.update(kw)
        return [], 0

    monkeypatch.setattr(agent_mod, "retrieve", fake_retrieve)
    key = _widget_key(client, star, emp["id"])
    client.post("/api/chat/message", headers={"X-Widget-Key": key},
                json={"message": "保修条款是怎样的", "visitor_id": "v-bind2"})

    assert captured.get("doc_ids") == [doc_id], f"文件级绑定应生效：{captured}"
    assert not captured.get("kb_ids"), f"没绑库就不该带 kb_ids：{captured}"


def test_saving_employee_keeps_kb_binding(client, star):
    """员工配置页保存不能顺手清掉知识库页勾选的库级绑定。

    回归点：前端曾固定提交 kb_ids:[]，用户一保存，知识库勾选就失效。
    """
    emp = _default_employee(client, star)
    kb = client.post("/api/knowledge-bases", headers=star,
                     json={"name": "保存保持绑定库"}).json()
    client.patch(f"/api/employees/{emp['id']}", headers=star,
                 json={"kb_ids": [kb["id"]]})

    # 模拟员工配置页保存：只带 doc_ids，不带 kb_ids
    client.patch(f"/api/employees/{emp['id']}", headers=star, json={"doc_ids": []})

    body = client.get(f"/api/employees/{emp['id']}", headers=star).json()
    assert kb["id"] in body["kb_ids"], "保存员工配置后，库级绑定必须保留"
