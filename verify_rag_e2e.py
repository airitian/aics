"""端到端验证：**真实 bge-m3** 走完「入库 → 检索 → 租户隔离」全链路。

为什么单独有这一个脚本：
- verify_embedding.py 只验证模型本身（通不通、维度、语义区分度）；
- verify_qdrant.py 验证 Qdrant 上的隔离（需要云端向量库可达）；
- 本脚本把真模型 + 真切分 + 真检索串起来，且**不依赖 Qdrant**
  （用 SQLite 向量后端），所以在云端向量库不可达时依然能证明
  「换了 bge-m3 之后检索召回是有效的」。

除向量库后端不同（db vs qdrant），其余代码路径与生产完全一致。

用法：
    python verify_rag_e2e.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent
_DB = _ROOT / "_verify_rag_e2e.db"

# 必须在导入 app 之前设置：用独立库，避免污染开发库
os.environ["DATABASE_URL"] = f"sqlite:///{_DB.as_posix()}"
os.environ["ENV"] = "dev"
os.environ["BOOTSTRAP_DEMO"] = "true"
os.environ["VECTOR_BACKEND"] = "db"  # 刻意不用 qdrant：本脚本要脱离云端向量库独立可跑
os.environ["LLM_PROVIDER"] = "stub"  # 只验证检索，不验证回答（LLM 还没接）

if _DB.exists():
    _DB.unlink()

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def banner(text: str) -> None:
    print(f"\n{text}")
    print("-" * 68)


def main() -> int:
    banner("0. 当前配置")
    print(f"  EMBED_PROVIDER = {settings.embed_provider}   <- 必须是 api")
    print(f"  EMBED_MODEL    = {settings.embed_model}  (dim={settings.embed_dim})")
    print(f"  VECTOR_BACKEND = {settings.vector_backend}")
    print(f"  TOP_K={settings.top_k}  MIN_SCORE={settings.min_score}")
    if settings.embed_provider != "api":
        print("\n  !! EMBED_PROVIDER 不是 api，本脚本无法验证真实模型。")
        return 2

    with TestClient(app) as c:
        banner("1. 启动自检：向量模型真的打通过一次")
        h = c.get("/healthz").json()
        emb = h.get("embed") or {}
        print(f"  embed = {emb}")
        check("启动自检里 embedding 为 ok", emb.get("ok") is True, str(emb.get("error")))
        check("自检回报的维度 == EMBED_DIM",
              emb.get("dim") == settings.embed_dim,
              f"{emb.get('dim')} vs {settings.embed_dim}")
        check("degraded 标志位存在", "degraded" in h, f"degraded={h.get('degraded')}")

        meta = c.get("/api/meta").json()
        check("meta 暴露真实向量模型名",
              meta["providers"]["embedding_model"] == settings.embed_model,
              meta["providers"]["embedding_model"])
        check("meta 暴露 rag 参数（top_k / min_score）",
              meta.get("rag", {}).get("min_score") == settings.min_score,
              str(meta.get("rag")))

        banner("2. 登录演示租户")
        def login(email):
            r = c.post("/api/auth/login", json={"email": email, "password": "demo12345"})
            assert r.status_code == 200, r.text
            return {"Authorization": "Bearer " + r.json()["access_token"]}

        star, sea = login("star@demo.local"), login("sea@demo.local")
        star_emp = c.get("/api/employees", headers=star).json()["items"][0]
        sea_emp = c.get("/api/employees", headers=sea).json()["items"][0]
        star_key = c.get(f"/api/employees/{star_emp['id']}/widget-keys", headers=star).json()["items"][0]["key"]
        sea_key = c.get(f"/api/employees/{sea_emp['id']}/widget-keys", headers=sea).json()["items"][0]["key"]
        check("两个租户各自拿到渠道凭证", bool(star_key and sea_key))

        banner("3. 真实 bge-m3 检索（bootstrap 的演示知识库已被真模型向量化）")
        t0 = time.perf_counter()
        chat = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
                      json={"message": "户外电源保修多久？", "visitor_id": "e2e-1"}).json()
        elapsed = (time.perf_counter() - t0) * 1000
        hits = chat.get("hits", [])
        scores = sorted((h.get("score") or 0 for h in hits), reverse=True)
        check("命中本租户知识（说明 MIN_SCORE 没把真答案拦掉）", bool(hits),
              f"命中 {len(hits)} 条，得分 {[round(s, 4) for s in scores]}")
        if hits:
            check("命中分数高于 MIN_SCORE",
                  min(scores) >= settings.min_score,
                  f"最低命中分 {min(scores):.4f} vs 门槛 {settings.min_score}")
            print(f"  Top 命中片段：{hits[0].get('text', '')[:70]}")
        print(f"  一次「向量化 + 检索 + 组装」耗时 {elapsed:.0f} ms（含一次真实 API 调用）")

        banner("4. 租户隔离：真模型下的跨租户知识不可互相召回")
        sea_chat = c.post("/api/chat/message", headers={"X-Widget-Key": sea_key},
                          json={"message": "老客户折扣码是多少 STAR20", "visitor_id": "e2e-2"}).json()
        sea_texts = " ".join(h["text"] for h in sea_chat.get("hits", []))
        check("海蓝检索不到星辰的机密（STAR20）", "STAR20" not in sea_texts,
              f"海蓝命中 {len(sea_chat.get('hits', []))} 条，均属自身知识")

        own = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
                     json={"message": "老客户折扣码 STAR20", "visitor_id": "e2e-3"}).json()
        own_texts = " ".join(h["text"] for h in own.get("hits", []))
        check("星辰能检索到自己的机密（反证：不是谁都查不到）", "STAR20" in own_texts,
              f"命中 {len(own.get('hits', []))} 条")

        banner("5. 无关问题不应硬塞知识（MIN_SCORE 门槛是否真的在拦噪音）")
        noise = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
                       json={"message": "今天北京天气怎么样", "visitor_id": "e2e-4"}).json()
        noise_scores = sorted((h.get("score") or 0 for h in noise.get("hits", [])), reverse=True)
        check("天气类无关问题没有召回高分段知识", not noise_scores or noise_scores[0] < 0.75,
              f"命中 {len(noise_scores)} 条，最高 {noise_scores[0]:.4f}" if noise_scores else "0 条")
        print(f"  无关问题的命中情况：{len(noise_scores)} 条，得分 "
              f"{[round(s, 4) for s in noise_scores] if noise_scores else '无'}")

        banner("6. 高优先级兜底不依赖检索")
        ho = c.post("/api/chat/message", headers={"X-Widget-Key": star_key},
                    json={"message": "你们这是诈骗，我要投诉！", "visitor_id": "e2e-5"}).json()
        check("投诉类强制转人工（与向量库/模型是否可用无关）", ho.get("handoff") is True,
              str(ho.get("handoff_reason", "")))

    banner("结果")
    print(f"  通过 {PASS} 项，失败 {FAIL} 项")
    # 临时库删不掉不影响结论，别让它把脚本变成非 0 退出
    try:
        if _DB.exists():
            _DB.unlink()
            print("  已清理临时数据库")
    except Exception as exc:  # noqa: BLE001
        print(f"  临时数据库未能删除（可手动删）：{_DB.name}  <- {exc}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
