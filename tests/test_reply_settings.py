"""AI 回复设置：延迟/拆分区间校验、旧单值字段回退、部分更新归一化。"""
from __future__ import annotations


def _default_employee(client, headers) -> dict:
    items = client.get("/api/employees", headers=headers).json()["items"]
    return next(e for e in items if e["is_default"])


def _patch(client, headers, emp_id: str, payload: dict):
    return client.patch(f"/api/employees/{emp_id}", headers=headers, json=payload)


def _get(client, headers, emp_id: str) -> dict:
    return next(e for e in client.get("/api/employees", headers=headers).json()["items"]
                if e["id"] == emp_id)


def _reset(client, headers, emp_id: str) -> None:
    _patch(client, headers, emp_id, {
        "reply_delay_seconds": 0,
        "reply_delay_min": 0, "reply_delay_max": 0,
        "split_interval_min_ms": 800, "split_interval_max_ms": 800,
        "split_reply_interval_ms": 800,
        "split_reply_enabled": False, "humanize_enabled": False,
    })


def test_delay_and_split_interval_validation(client, star):
    emp = _default_employee(client, star)
    # 区间两端同时给出时，min > max 必须被拒
    assert _patch(client, star, emp["id"], {"reply_delay_min": 5, "reply_delay_max": 2}).status_code == 422
    assert _patch(client, star, emp["id"],
                  {"split_interval_min_ms": 3000, "split_interval_max_ms": 1000}).status_code == 422
    # 合法区间可以保存并回读
    ok = _patch(client, star, emp["id"], {"reply_delay_min": 2, "reply_delay_max": 5})
    assert ok.status_code == 200, ok.text
    got = _get(client, star, emp["id"])
    assert got["reply_delay_min"] == 2 and got["reply_delay_max"] == 5
    _reset(client, star, emp["id"])


def test_legacy_single_value_fallback(client, star):
    """只更新旧单值字段（模拟旧数据）时，序列化应回退成等值区间。"""
    emp = _default_employee(client, star)
    try:
        assert _patch(client, star, emp["id"], {"reply_delay_seconds": 5}).status_code == 200
        got = _get(client, star, emp["id"])
        assert got["reply_delay_min"] == 5 and got["reply_delay_max"] == 5, \
            "旧固定延迟应回退为 min=max=5 的区间"

        assert _patch(client, star, emp["id"], {"split_reply_interval_ms": 2000}).status_code == 200
        got = _get(client, star, emp["id"])
        assert got["split_interval_min_ms"] == 2000 and got["split_interval_max_ms"] == 2000, \
            "旧固定间隔应回退为 min=max=2000 的区间"
    finally:
        _reset(client, star, emp["id"])


def test_partial_update_single_end_normalizes(client, star):
    """部分更新只给一端时（另一端还是旧值），应用后自动交换归一，不落脏状态。"""
    emp = _default_employee(client, star)
    try:
        assert _patch(client, star, emp["id"], {"reply_delay_min": 10}).status_code == 200
        got = _get(client, star, emp["id"])
        assert got["reply_delay_min"] <= got["reply_delay_max"], \
            "部分更新后不允许出现 min>max 的区间"
        assert (got["reply_delay_min"], got["reply_delay_max"]) == (0, 10)

        assert _patch(client, star, emp["id"], {"split_interval_max_ms": 300}).status_code == 200
        got = _get(client, star, emp["id"])
        assert got["split_interval_min_ms"] <= got["split_interval_max_ms"]
        assert (got["split_interval_min_ms"], got["split_interval_max_ms"]) == (300, 800)
    finally:
        _reset(client, star, emp["id"])
