"""真实规模提示词下的分阶段耗时（检索命中数、提示词大小都是真实值）。

与 profile_latency.py 的区别：C 项那里用了 104 token 的小提示词，
这里用 build_system_prompt 拼出真实 prompt（含 top_k=5 段知识、记忆、历史）再计时。
只读（persist=False），不写库。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import requests

BASE = os.environ.get("AICS_BASE", "http://127.0.0.1:8000")
EMAIL = os.environ.get("AICS_EMAIL", "star@demo.local")
PASSWORD = os.environ.get("AICS_PASSWORD", "demo12345")
sys.path.insert(0, ".")


async def main() -> int:
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=20)
    tok = r.json()["access_token"]
    h = {"Authorization": "Bearer " + tok}
    me = s.get(f"{BASE}/api/auth/me", headers=h, timeout=10).json()
    tenant_id = me["tenant"]["id"]
    emp_id = s.get(f"{BASE}/api/employees", headers=h, timeout=10).json()["items"][0]["id"]

    from sqlalchemy import select
    from app.database import SessionLocal
    from app.models import AiEmployee, Session as ChatSession, Tenant
    from app import prompting
    from app.agent import _recent_history, kb_ids_for_employee
    from app.rag import dedupe_hits, retrieve
    from app.llm import chat

    db = SessionLocal()
    tenant = db.get(Tenant, tenant_id)
    employee = db.get(AiEmployee, emp_id)
    sess = db.execute(
        select(ChatSession).where(ChatSession.tenant_id == tenant_id, ChatSession.employee_id == emp_id)
        .order_by(ChatSession.created_at.desc()).limit(1)
    ).scalars().first()
    print(f"tenant={tenant_id} emp={emp_id} session={sess.id if sess else None}")

    QUESTION = "亚马逊供应链标准对审计有什么要求"
    for rnd in range(3):
        t0 = time.perf_counter()
        hits, embed_tokens = await retrieve(db, tenant_id=tenant_id,
                                            kb_ids=kb_ids_for_employee(db, tenant_id, emp_id),
                                            query=QUESTION)
        hits = dedupe_hits(hits)
        t1 = time.perf_counter()
        history = _recent_history(db, tenant.id, sess.id) if sess else []
        memory = [] if not employee.memory_enabled else []
        sp = prompting.build_system_prompt(employee=employee, tenant_name=tenant.name,
                                           hits=hits, now=__import__("app.utils", fromlist=["utcnow"]).utcnow(),
                                           visitor_language="zh-CN", memory_lines=memory)
        msgs = prompting.build_messages(sp, history, QUESTION)
        prompt_chars = len(sp)
        t2 = time.perf_counter()
        lr = await chat(msgs, temperature=employee.llm_temperature / 100.0,
                        max_tokens=employee.llm_max_tokens)
        t3 = time.perf_counter()
        print(f"第{rnd+1}轮 检索 {(t1-t0)*1000:5.0f}ms (hits={len(hits)}) | "
              f"拼prompt {(t2-t1)*1000:5.0f}ms ({prompt_chars}字) | "
              f"LLM {(t3-t2)*1000:5.0f}ms (prompt={lr.prompt_tokens} completion={lr.completion_tokens} "
              f"答{len(lr.text)}字)", flush=True)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
