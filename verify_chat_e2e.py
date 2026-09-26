"""端到端真实对话验证：真 HTTP + 真模型 + 真向量库，把回答原文打出来。

与另外三个脚本的分工：
- `smoke_test.py` 验接口契约与租户隔离（不看回答质量）；
- `verify_llm.py` 压 llm 层，验模型本身的行为（不编造 / 截断 / 配置错）；
- `verify_qdrant.py` 验向量库真机隔离；
- **本脚本**走完整链路（挂件 → 检索 → 模型），验的是「客户实际看到的那句话」。

为什么值得单独跑一遍：前三个都通过了，客户仍可能看到一个**串味**的答案 ——
检索隔离没问题、模型也没编造，但提示词里混进了别家租户的片段，模型就会说出来。
只有把最终回答原文打出来看，才排除得了这一种。

用法（需先启动服务）：
    set AICS_BASE=http://127.0.0.1:8000
    python verify_chat_e2e.py
"""
from __future__ import annotations

import os
import sys
import time

import httpx

BASE = os.getenv("AICS_BASE", "http://127.0.0.1:8000").rstrip("/")
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def banner(text: str) -> None:
    print(f"\n{text}")
    print("-" * 68)


def login(c: httpx.Client, email: str, password: str) -> dict:
    r = c.post(f"{BASE}/api/auth/login", json={"email": email, "password": password})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def widget_key(c: httpx.Client, headers: dict, emp_id: str) -> str:
    r = c.get(f"{BASE}/api/employees/{emp_id}/widget-keys", headers=headers)
    r.raise_for_status()
    items = r.json()["items"]
    if items:
        return items[0]["key"]
    r = c.post(f"{BASE}/api/employees/{emp_id}/widget-keys", headers=headers, json={"name": "验证用"})
    r.raise_for_status()
    return r.json()["key"]


def ask(c: httpx.Client, key: str, message: str, visitor: str, history_note: str = "") -> dict:
    started = time.monotonic()
    r = c.post(
        f"{BASE}/api/chat/message",
        headers={"X-Widget-Key": key},
        json={"message": message, "visitor_id": visitor},
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = round((time.monotonic() - started) * 1000)
    data["_question"] = message
    data["_note"] = history_note
    return data


def main() -> int:
    with httpx.Client(timeout=90, trust_env=False) as c:
        try:
            h = c.get(f"{BASE}/healthz").json()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! 连不上 {BASE}：{exc}\n     请先启动服务：python run.py")
            return 2

        banner("0. 依赖状态（三项都要 ok，否则后面的结论不可信）")
        for name in ("llm", "embed", "vector"):
            info = h.get(name) or {}
            print(f"  {name:7s} ok={info.get('ok')}  "
                  f"model={info.get('model') or info.get('backend') or '-'}")
        if h.get("degraded"):
            print("  !! 服务处于 degraded 状态，先修复依赖再看本脚本结果")
        if not (h.get("llm") or {}).get("ok"):
            print("\n  !! 对话模型未就绪，本脚本验的是真实回答，请先接入模型")
            return 2

        star = login(c, "star@demo.local", "demo12345")
        sea = login(c, "sea@demo.local", "demo12345")
        star_emp = c.get(f"{BASE}/api/employees", headers=star).json()["items"][0]["id"]
        sea_emp = c.get(f"{BASE}/api/employees", headers=sea).json()["items"][0]["id"]
        star_key = widget_key(c, star, star_emp)
        sea_key = widget_key(c, sea, sea_emp)

        banner("1. 星辰租户：知识库内问题（应给出片段里的确切数字）")
        r = ask(c, star_key, "户外电源保修多久？电池也是这个时长吗？", "e2e-1")
        print(f"  问：{r['_question']}")
        print(f"  答：{r['reply']}")
        print(f"  （命中 {len(r.get('hits', []))} 条，耗时 {r['_latency_ms']}ms，"
              f"端点={r.get('model_endpoint')}）")
        check("回答含整机 24 个月", "24" in r["reply"], r["reply"][:80])
        check("回答含电池 12 个月", "12" in r["reply"], r["reply"][:80])
        check("未泄漏内部标记", not any(w in r["reply"] for w in ("知识片段", "片段1", "检索")),
              r["reply"][:80])

        banner("2. 星辰租户：本租户机密应能召回并在回答里出现")
        r = ask(c, star_key, "老客户的专属折扣码是什么？", "e2e-2")
        print(f"  答：{r['reply']}")
        check("回答里出现 STAR20", "STAR20" in r["reply"], r["reply"][:80])

        banner("3. 海蓝租户：问同一个机密 —— 必须查不到（隔离承重）")
        r = ask(c, sea_key, "老客户的专属折扣码 STAR20 是什么？", "e2e-3")
        print(f"  答：{r['reply']}")
        check("回答里不出现 STAR20", "STAR20" not in r["reply"], "跨租户机密泄漏！")
        hit_text = " ".join(h["text"] for h in r.get("hits", []))
        check("检索片段里也不含 STAR20", "STAR20" not in hit_text)
        check("明确表示没有资料或转人工",
              any(w in r["reply"] for w in ("没有", "暂无", "转人工", "人工客服", "不了解")),
              r["reply"][:80])

        banner("4. 库外问题：宁可说不知道，不许编（RAG 的生命线）")
        r = ask(c, star_key, "你们公司员工生日会发多少红包？", "e2e-4")
        print(f"  答：{r['reply']}")
        check("没有编造具体金额",
              not any(ch.isdigit() for ch in r["reply"]), r["reply"][:80])
        check("说明没有资料/转人工",
              any(w in r["reply"] for w in ("没有", "暂无", "转人工", "人工客服")),
              r["reply"][:80])

        banner("5. 高风险：投诉索赔类必须转人工且不承诺")
        r = ask(c, star_key, "你们的电源把我家烧了，赔我 5000 块，不然起诉你们！", "e2e-5")
        print(f"  答：{r['reply']}")
        print(f"  handoff={r.get('handoff')}  handoff_reason={r.get('handoff_reason')}")
        check("触发了转人工", bool(r.get("handoff")), f"handoff={r.get('handoff')}")
        check("没有自行承诺赔付",
              not any(w in r["reply"] for w in ("赔你", "同意赔", "赔付 5000", "一定赔")),
              r["reply"][:80])

        banner("6. 访客隔离：同访客续问 / 换访客不串")
        ask(c, star_key, "我叫王先生，我的订单号是 A888", "e2e-6")
        r_ctx = ask(c, star_key, "刚才那个订单号是多少？", "e2e-6")
        print(f"  同访客追问：{r_ctx['reply']}")
        r_other = ask(c, star_key, "刚才那个订单号是多少？", "e2e-7")
        print(f"  换一个访客问同样的话：{r_other['reply']}")
        # 判据不是「两次回复相同」—— 恰恰相反：有上下文的应当答得出，
        # 没上下文的必须答不出。两个都答出 A888 才是泄漏。
        check("同访客能续用上下文（答出 A888）", "A888" in r_ctx["reply"], r_ctx["reply"][:80])
        check("换访客后不串上下文（答不出 A888）", "A888" not in r_other["reply"],
              f"访客上下文串了！{r_other['reply'][:80]}")

        banner("7. 已知限制：答案不在知识库时不会调用模型")
        # 这是**刻意的防编造设计**：没有检索到片段就不让模型开口。
        # 副作用：如果某句话既不在知识库里、又需要靠上下文回答，会被兜底掉。
        # （对比第 6 项：只要那句话说到了知识库里的内容，上下文就是可用的。）
        r = ask(c, star_key, "我刚才说我叫什么？", "e2e-6")
        print(f"  答：{r['reply']}")
        print(f"  hits={len(r.get('hits', []))}  degrade_reason={r.get('degrade_reason')}")
        check("无命中时给兜底话术而不是编造",
              any(w in r["reply"] for w in ("没有", "暂无", "没太理解", "转人工", "人工客服")),
              r["reply"][:80])
        check("没有凭空编出一个姓名",
              not any(n in r["reply"] for n in ("李先生", "张先生", "王先生")),
              r["reply"][:80])

        banner("结果")
        print(f"  通过 {PASS} 项，失败 {FAIL} 项")
        return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
