"""自动停止回复：条件组（组内且 / 组间或）、旧字段回退、入参校验。

参考 SaleSmartly 的「自动停止回复」交互：条件组内条件需同时满足（且），
满足任一条件组即触发停止（或）；触发时先发结束语（可空），本会话后续
消息不再回复；用户生成新会话后恢复正常接待。
"""
from __future__ import annotations


def _default_employee(client, headers) -> dict:
    items = client.get("/api/employees", headers=headers).json()["items"]
    return next(e for e in items if e["is_default"])


def _patch(client, headers, emp_id: str, payload: dict):
    return client.patch(f"/api/employees/{emp_id}", headers=headers, json=payload)


def _reset(client, headers, emp_id: str) -> None:
    _patch(client, headers, emp_id, {
        "stop_reply_enabled": False,
        "stop_condition_groups": [],
        "stop_reply_rounds": 0,
        "stop_reply_message": "",
        "stop_time_enabled": False,
        "stop_time_rules": [],
    })


def _new_session(client, headers, emp_id: str) -> str:
    return client.post(f"/api/testchat/{emp_id}/sessions", headers=headers).json()["id"]


def _send(client, headers, emp_id: str, sid: str, msg: str):
    return client.post(
        f"/api/testchat/{emp_id}/sessions/{sid}/messages", headers=headers, json={"message": msg}
    )


# --------------------------------------------------------------------------- #
# 入参校验
# --------------------------------------------------------------------------- #
def test_stop_groups_validation(client, star):
    emp = _default_employee(client, star)
    bad_cases = [
        ("空条件组", {"stop_condition_groups": [[]]}),
        ("未知条件类型", {"stop_condition_groups": [[{"type": "nope", "op": "gte", "value": 3}]]}),
        ("不支持的比较符", {"stop_condition_groups": [[{"type": "ai_rounds", "op": "gt", "value": 3}]]}),
        ("阈值为 0", {"stop_condition_groups": [[{"type": "ai_rounds", "op": "gte", "value": 0}]]}),
        ("阈值超上限", {"stop_condition_groups": [[{"type": "ai_rounds", "op": "gte", "value": 1000}]]}),
    ]
    for label, payload in bad_cases:
        res = _patch(client, star, emp["id"], payload)
        assert res.status_code == 422, f"{label} 应被拒绝：{res.text}"

    ok = _patch(client, star, emp["id"], {
        "stop_condition_groups": [[{"type": "ai_rounds", "op": "gte", "value": 3}]],
    })
    assert ok.status_code == 200, ok.text
    got = next(e for e in client.get("/api/employees", headers=star).json()["items"]
               if e["id"] == emp["id"])
    assert got["stop_condition_groups"] == [[{"type": "ai_rounds", "op": "gte", "value": 3}]]
    _reset(client, star, emp["id"])


# --------------------------------------------------------------------------- #
# 触发与恢复
# --------------------------------------------------------------------------- #
def test_ai_rounds_group_stops_and_new_session_recovers(client, star):
    emp = _default_employee(client, star)
    try:
        assert _patch(client, star, emp["id"], {
            "stop_reply_enabled": True,
            "stop_condition_groups": [[{"type": "ai_rounds", "op": "gte", "value": 1}]],
            "stop_reply_message": "感谢咨询，再见",
        }).status_code == 200

        sid = _new_session(client, star, emp["id"])
        r1 = _send(client, star, emp["id"], sid, "你们支持货到付款吗")
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["reply"] and body1["reply"] != "感谢咨询，再见", "第一轮 AI 尚未回复过，不应触发停止"
        assert body1["status"] != "stopped"

        r2 = _send(client, star, emp["id"], sid, "那大概几天能到")
        body2 = r2.json()
        assert body2["reply"] == "感谢咨询，再见", "达到阈值那条消息应先收到结束语"
        assert body2["status"] == "stopped"

        r3 = _send(client, star, emp["id"], sid, "还在吗")
        body3 = r3.json()
        assert body3["status"] == "stopped", "已停止的会话应保持停止状态"
        assert body3["reply"] == "", "停止后不应再产生任何回复"

        # 新会话自动恢复接待
        sid2 = _new_session(client, star, emp["id"])
        r4 = _send(client, star, emp["id"], sid2, "你们支持货到付款吗")
        assert r4.json()["status"] != "stopped"
        assert r4.json()["reply"], "新会话应正常接待"
    finally:
        _reset(client, star, emp["id"])


def test_group_and_or_semantics(client, star):
    """组内条件「且」需全部满足；组间「或」任一组满足即停止。"""
    emp = _default_employee(client, star)
    try:
        # 组1：AI 轮次>=999（永不满足）；组2：访客消息>=2。靠组2触发，验证组间或。
        assert _patch(client, star, emp["id"], {
            "stop_reply_enabled": True,
            "stop_condition_groups": [
                [{"type": "ai_rounds", "op": "gte", "value": 999}],
                [{"type": "visitor_msgs", "op": "gte", "value": 2}],
            ],
            "stop_reply_message": "本次服务到此结束",
        }).status_code == 200

        sid = _new_session(client, star, emp["id"])
        r1 = _send(client, star, emp["id"], sid, "你们支持货到付款吗")
        assert r1.json()["status"] != "stopped", "仅 1 条访客消息，不应触发"

        r2 = _send(client, star, emp["id"], sid, "那大概几天能到")
        body2 = r2.json()
        assert body2["reply"] == "本次服务到此结束"
        assert body2["status"] == "stopped"

        # 组内「且」：AI 轮次条件不满足时，即使访客消息条数达标也不停止
        assert _patch(client, star, emp["id"], {
            "stop_reply_enabled": True,
            "stop_condition_groups": [
                [{"type": "ai_rounds", "op": "gte", "value": 999},
                 {"type": "visitor_msgs", "op": "gte", "value": 1}],
            ],
            "stop_reply_message": "不应出现",
        }).status_code == 200
        sid2 = _new_session(client, star, emp["id"])
        r3 = _send(client, star, emp["id"], sid2, "你们支持货到付款吗")
        body3 = r3.json()
        assert body3["status"] != "stopped", "组内另一条件未满足，不应触发停止"
        assert body3["reply"] != "不应出现"
    finally:
        _reset(client, star, emp["id"])


def test_legacy_rounds_fallback(client, star):
    """旧数据只有 stop_reply_rounds 时，回退为单组「AI轮次>=N」条件。"""
    emp = _default_employee(client, star)
    try:
        assert _patch(client, star, emp["id"], {
            "stop_reply_enabled": True,
            "stop_condition_groups": [],
            "stop_reply_rounds": 2,
            "stop_reply_message": "回退生效",
        }).status_code == 200

        sid = _new_session(client, star, emp["id"])
        r1 = _send(client, star, emp["id"], sid, "你们支持货到付款吗")
        assert r1.json()["status"] != "stopped", "第 1 轮后未达阈值"
        r2 = _send(client, star, emp["id"], sid, "那大概几天能到")
        assert r2.json()["status"] != "stopped", "第 2 轮检查时 AI 只回复过 1 次"
        r3 = _send(client, star, emp["id"], sid, "还在吗")
        body3 = r3.json()
        assert body3["reply"] == "回退生效"
        assert body3["status"] == "stopped"
    finally:
        _reset(client, star, emp["id"])
