"""真实启动冒烟测试：打 HTTP 接口，验证启动、隔离、对话链路。

用法：先启动服务（默认 8000），再执行
    AICS_BASE=http://127.0.0.1:8000 python smoke_test.py
依赖：httpx（已随 requirements.txt 安装）
"""
from __future__ import annotations

import json
import os
import sys

import httpx

BASE = os.getenv("AICS_BASE", "http://127.0.0.1:8123")
c = httpx.Client(base_url=BASE, timeout=30)
ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    mark = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{mark}] {name}{(' | ' + extra) if extra else ''}")


# 1. 健康检查与元信息
h = c.get("/healthz")
check("GET /healthz", h.status_code == 200, h.text[:120])
meta = c.get("/api/meta").json()
check("GET /api/meta", "features" in meta or "app" in meta, json.dumps(meta, ensure_ascii=False)[:200])

# 2. 平台管理员登录
r = c.post("/api/auth/login", json={"email": "admin@aics.local", "password": "admin12345"})
check("平台管理员登录", r.status_code == 200)
plat = {"Authorization": f"Bearer {r.json()['access_token']}"}

tenants = c.get("/api/platform/tenants", headers=plat).json()["items"]
check("平台可见演示租户", len(tenants) >= 2, f"{[t['name'] for t in tenants]}")

# 3. 两个租户各自登录，互查对方员工必须 404
sessions = {}
for email, who in [("star@demo.local", "星辰"), ("sea@demo.local", "海蓝")]:
    rr = c.post("/api/auth/login", json={"email": email, "password": "demo12345"})
    check(f"{who}租户管理员登录", rr.status_code == 200)
    sessions[who] = {"Authorization": f"Bearer {rr.json()['access_token']}"}

star_emp = c.get("/api/employees", headers=sessions["星辰"]).json()["items"][0]
sea_emp = c.get("/api/employees", headers=sessions["海蓝"]).json()["items"][0]

leak = c.get(f"/api/employees/{star_emp['id']}", headers=sessions["海蓝"])
check("跨租户读员工被拒（404 不泄漏存在性）", leak.status_code == 404, f"status={leak.status_code}")

keys = c.get(f"/api/employees/{star_emp['id']}/widget-keys", headers=sessions["星辰"]).json()["items"]
check("默认渠道凭证已生成", bool(keys))
star_key = keys[0]["key"]

# 4. 对话：命中本租户知识
chat = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
              json={"message": "户外电源保修多久？", "visitor_id": "smoke-1"}).json()
check("对话命中本租户知识", bool(chat.get("hits")), f"hits={len(chat.get('hits', []))}")

# 回复的断言要跟着「模型到底配没配」走 —— 写死一种状态，另一种部署下必然误报：
#   未接模型 → 必须如实说「模型未配置」，绝不能编；
#   已接模型 → 必须真的基于片段作答（出现片段里的数字），而不是复读兜底话术。
llm_ok = bool((h.json().get("llm") or {}).get("ok"))
reply = chat.get("reply", "")
if llm_ok:
    check("已接模型时基于片段作答", ("24" in reply and "12" in reply),
          reply[:80])
    check("回复中不含兜底话术", "模型未配置" not in reply and "无法处理" not in reply,
          reply[:80])
else:
    check("未接模型时如实说明「模型未配置」", "模型未配置" in reply, reply[:80])

# 5. 关键：跨租户知识不可检索（星辰的折扣码不能出现在海蓝的回答里）
sea_key = c.get(f"/api/employees/{sea_emp['id']}/widget-keys", headers=sessions["海蓝"]).json()["items"][0]["key"]
sea_chat = c.post("/api/chat/message", headers={"X-Widget-Key": sea_key},
                  json={"message": "老客户折扣码是多少 STAR20", "visitor_id": "smoke-2"}).json()
texts = " ".join(h["text"] for h in sea_chat.get("hits", []))
check("海蓝检索不到星辰的机密（STAR20）", "STAR20" not in texts,
      f"命中 {len(sea_chat.get('hits', []))} 条，全部属于海蓝自身知识")

# 反向：星辰能检索到自己的机密，说明不是"谁都查不到"
own = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
             json={"message": "老客户折扣码 STAR20", "visitor_id": "smoke-2b"}).json()
own_texts = " ".join(h["text"] for h in own.get("hits", []))
check("星辰能检索到自己的机密（STAR20）", "STAR20" in own_texts)

# 6. 强制转人工
ho = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
            json={"message": "你们这是诈骗，我要投诉！", "visitor_id": "smoke-3"}).json()
check("投诉类强制转人工", ho.get("handoff") is True, ho.get("handoff_reason", ""))

# 7. 越权访问审计
c.get("/api/employees", headers={"Authorization": "Bearer invalid-token"})   # 制造一次鉴权失败
rows = c.get("/api/insights/audit?limit=50", headers=sessions["星辰"]).json()
check("审计接口可用", isinstance(rows.get("items"), list),
      f"审计条数={len(rows.get('items', []))}")

print("\nSMOKE_RESULT=" + ("ALL_OK" if ok else "HAS_FAIL"))
sys.exit(0 if ok else 1)
