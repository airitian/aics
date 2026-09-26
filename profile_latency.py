"""对话回复耗时剖析：分阶段实测线上真实依赖的延迟（只读，不改数据）。

阶段：
  A. embedding（bge-m3 @ Gitee AI）—— query 向量化
  B. Qdrant 检索（sa-east-1）—— 向量搜索
  C. LLM 生成（deepseek-v4-flash @ PackyAPI，非流式）
  E. 端到端 HTTP（走 testchat 真实接口，含 DB 与全部业务逻辑）

A/B/C 各跑 3 轮：第 1 轮含 TLS/连接建立，后 2 轮为连接复用后的稳态。
用法：python profile_latency.py   （需服务已在 127.0.0.1:8000 运行；在 aics 目录下执行）
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

QUERY = "亚马逊供应链标准对审计有什么要求"
ROUNDS = 3
sys.path.insert(0, ".")


def report(stage: str, times_ms: list[float], extra: str = "") -> None:
    line = f"{stage:<14} " + "  ".join(f"第{i+1}轮 {t:7.0f}ms" for i, t in enumerate(times_ms))
    if extra:
        line += f"   {extra}"
    print(line, flush=True)


async def main() -> int:
    # ---- 登录拿 token / tenant / employee ----
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=20)
    assert r.status_code == 200, r.text[:300]
    tok = r.json()["access_token"]
    h = {"Authorization": "Bearer " + tok}
    me = s.get(f"{BASE}/api/auth/me", headers=h, timeout=10).json()
    tenant_id = me["tenant"]["id"]
    emps = s.get(f"{BASE}/api/employees", headers=h, timeout=10).json()
    emp_id = emps["items"][0]["id"]
    print(f"tenant={tenant_id} employee={emp_id}")
    print("-" * 92, flush=True)

    from app.config import settings
    from app.embedding import embed_texts
    from app.llm import chat
    from app.vectorstore.qdrant_store import QdrantVectorStore

    print(f"embed: {settings.embed_model or settings.embed_provider}  llm: {settings.llm_model}")
    print(f"qdrant: {settings.qdrant_url}   top_k={settings.top_k} min_score={settings.min_score}")
    print("-" * 92, flush=True)

    # ---- A. embedding ----
    times, res = [], []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        emb = await embed_texts([QUERY])
        times.append((time.perf_counter() - t0) * 1000)
        res.append(emb)
    report("A.embedding", times, f"dim={len(res[-1].vectors[0])} tokens={res[-1].tokens}")
    vec = res[-1].vectors[0]

    # ---- B. qdrant search ----
    store = QdrantVectorStore(dim=len(vec))
    times, hits_all = [], []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        hits = store.search(tenant_id, vec, top_k=settings.top_k, min_score=0.0)
        times.append((time.perf_counter() - t0) * 1000)
        hits_all.append(hits)
    top = hits_all[-1][0].score if hits_all[-1] else 0
    report("B.qdrant", times, f"hits={len(hits_all[-1])} top={top:.4f}")

    # ---- C. LLM 生成 ----
    msgs = [{"role": "system", "content": "你是客服助手，用中文简洁回答。"},
            {"role": "user", "content": QUERY + "？请简要回答。"}]
    times = []
    last = None
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        last = await chat(msgs, temperature=0.3, max_tokens=1200)
        times.append((time.perf_counter() - t0) * 1000)
    report("C.llm", times,
           f"prompt={last.prompt_tokens} completion={last.completion_tokens} "
           f"回答{len(last.text)}字 endpoint={last.endpoint}")

    # ---- E. 端到端 HTTP（真实 testchat 接口）----
    sid = s.post(f"{BASE}/api/testchat/{emp_id}/sessions", headers=h, timeout=10).json()["id"]
    times = []
    for i in range(ROUNDS):
        t0 = time.perf_counter()
        r = s.post(f"{BASE}/api/testchat/{emp_id}/sessions/{sid}/messages", headers=h,
                   json={"message": f"质检文件需要保存多久？{i}"}, timeout=120)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text[:300]
    reply = r.json()
    report("E.e2e-http", times,
           f"top={reply.get('top_score', 0):.4f} conf={reply.get('confidence')} 回复{len(reply.get('reply', ''))}字")
    s.delete(f"{BASE}/api/testchat/{emp_id}/sessions", headers=h, timeout=10)
    print("-" * 92)
    print("说明：A/B/C 第1轮含建连开销；对比第2/3轮看稳态。E 是用户真实感受到的总耗时。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
