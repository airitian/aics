"""AI 员工测试（聊天式）真机验证。

跑的是完整真实链路：真实对话模型 + 真实向量库 + 真实知识库内容。
需要服务已在 127.0.0.1:8000 运行，且 star 租户已入库《Amazon 供应链标准手册》。

验证点（每条都对应一种「看起来能用、其实没接上」的失败）：
1. 多轮链路可用 —— 同一会话里连续多轮，检索与生成都正常（历史在 `_recent_history`）。
2. 上下文指代类提问的行为 —— 见 `[4]`：本版本「无知识命中就不调模型」，
   所以「我刚才问了什么」这类问题会被兜底掉。这是刻意的防编造设计
   （README 第 8 节），不是上下文丢失，但测试时必须知道，否则会误判成 bug。
3. 兜底不编造 —— 问手册里没有的东西，正确行为是承认查不到。
4. 历史完整 —— AI 的兜底回复也要落库，刷新后不能凭空消失。
5. 隔离 —— 测试会话不进概览统计、不进坐席会话列表。
6. 清理 —— 删除测试会话后，消息一并删除。
"""
from __future__ import annotations

import os
import re
import sys

import requests

BASE = os.environ.get("AICS_BASE", "http://127.0.0.1:8000")
EMAIL = os.environ.get("AICS_EMAIL", "star@demo.local")
PASSWORD = os.environ.get("AICS_PASSWORD", "demo12345")

FALLBACK_HINTS = ("没有", "未", "查不到", "找不到", "暂无", "不清楚", "无法", "转接人工", "转人工", "人工客服")
PREV_TOPIC_HINTS = ("手册", "供应链", "更新", "2023", "时间", "版本", "标准")

PASS = 0
FAIL = 0
NOTES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))
    return cond


def note(text: str) -> None:
    NOTES.append(text)
    print(f"  [INFO] {text}")


def main() -> int:
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=20)
    if r.status_code != 200:
        print(f"登录失败 {r.status_code}: {r.text[:200]}")
        return 2
    h = {"Authorization": "Bearer " + r.json()["access_token"]}

    health = s.get(f"{BASE}/healthz", timeout=30).json()
    print(f"\n依赖状态：degraded={health['degraded']} "
          f"llm={health['llm'].get('model')} embed={health['embed'].get('model')} "
          f"vector={health['vector'].get('backend')} points={health['vector'].get('points')}")
    if health["degraded"]:
        print("警告：外部依赖处于降级状态，下面的结果可能不代表正常表现")

    emp = next(e for e in s.get(f"{BASE}/api/employees", headers=h, timeout=20).json()["items"]
               if e["is_default"])
    kbs = s.get(f"{BASE}/api/knowledge-bases", headers=h, timeout=20).json()["items"]
    print(f"被测员工：{emp['name']}（绑定 {len(emp['kb_ids'])} 个知识库）"
          f"　可用知识库：{', '.join(k['name'] for k in kbs)}")

    overview_before = s.get(f"{BASE}/api/insights/overview?days=7", headers=h, timeout=30).json()

    # ---------------------------------------------------------------- 准备会话
    print("\n[1] 创建测试会话")
    r = s.post(f"{BASE}/api/testchat/{emp['id']}/sessions", headers=h, timeout=20)
    check(r.status_code == 200, "创建测试会话", r.text[:200])
    sid = r.json()["id"]
    print(f"        session_id={sid}")

    def ask(text: str) -> dict:
        res = s.post(f"{BASE}/api/testchat/{emp['id']}/sessions/{sid}/messages",
                     headers=h, json={"message": text}, timeout=180)
        if res.status_code != 200:
            print(f"        请求失败 {res.status_code}: {res.text[:200]}")
            return {}
        return res.json()

    try:
        # ------------------------------------------------------------ 第 1 轮
        print("\n[2] 第一轮：文档内问题（应命中知识库）")
        q1 = "这本《供应链标准手册》是什么时候更新的？"
        r1 = ask(q1)
        print(f"        Q: {q1}\n        A: {r1.get('reply','')[:160]}")
        note(f"第 1 轮：命中 {len(r1.get('hits', []))} 段，最高相关度 {r1.get('top_score')}，"
             f"{r1.get('latency_ms')} ms，{r1.get('tokens')} tokens")
        check(bool(r1.get("reply")), "第一轮有回复")
        check(len(r1.get("hits", [])) > 0, "第一轮命中知识库片段",
              "未命中 → 检查知识库是否已绑定到该员工")

        # ------------------------------------------------------------ 第 2 轮
        print("\n[3] 第二轮：带历史的同主题追问（多轮链路仍应正常检索与生成）")
        q2 = "手册里要求供应商遵守哪些方面的标准？"
        r2 = ask(q2)
        print(f"        Q: {q2}\n        A: {r2.get('reply','')[:160]}")
        note(f"第 2 轮：命中 {len(r2.get('hits', []))} 段，最高相关度 {r2.get('top_score')}，"
             f"{r2.get('latency_ms')} ms，{r2.get('tokens')} tokens")
        check(bool(r2.get("reply")), "第二轮有回复（会话未中断）")
        check(len(r2.get("hits", [])) > 0, "第二轮仍能命中知识库")
        if r1.get("tokens") and r2.get("tokens"):
            note(f"tokens 对比：第 1 轮 {r1['tokens']} → 第 2 轮 {r2['tokens']}"
                 f"（第 2 轮提示词里多了前几轮对话，通常会更长）")

        # ------------------------------------------------------------ 第 3 轮
        print("\n[4] 第三轮：上下文指代类提问（本版本预期行为见下）")
        q3 = "我上一个问题问的是什么？请用一句话重复。"
        r3 = ask(q3)
        reply3 = r3.get("reply", "")
        print(f"        Q: {q3}\n        A: {reply3[:200]}")
        is_fallback = any(k in reply3 for k in FALLBACK_HINTS)
        mentions_prev = any(k in reply3 for k in PREV_TOPIC_HINTS)
        check(is_fallback or mentions_prev,
              "上下文指代问题：要么据实回答上一轮、要么老实兜底，没有编造",
              f"回复既没兜底也没提及上一轮：{reply3[:120]}")
        if is_fallback and not mentions_prev:
            note("「我刚才问了什么」被兜底了 —— 这是「无知识命中就不调模型」的刻意防编造设计"
                 "（README 第 8 节），不是上下文丢失：模型确实拿到了历史，只是这句话没沾知识库，"
                 "系统按设计直接兜底。测试时请知悉：纯上下文类提问不会得到回答。")

        # ------------------------------------------------------------ 第 4 轮
        print("\n[5] 第四轮：文档外问题（应承认查不到，而不是编造）")
        q4 = "贵公司今年双十一的促销折扣是多少？"
        r4 = ask(q4)
        reply4 = r4.get("reply", "")
        print(f"        Q: {q4}\n        A: {reply4[:200]}")
        check(any(k in reply4 for k in FALLBACK_HINTS),
              "面对文档外问题承认查不到/转人工",
              f"未识别为兜底话术：{reply4[:160]}")
        # 只认「具体折扣数值」，不能因为复述问题里的「折扣」二字就误判成编造
        fabricated = bool(re.search(r"\d+\s*折", reply4)) or bool(re.search(r"\d+\s*%", reply4))
        check(not fabricated, "回复中没有出现具体的折扣数值", f"疑似给出数值：{reply4[:160]}")

        # ------------------------------------------------------------ 历史
        print("\n[6] 历史回放：AI 的回复必须都在（含兜底分支）")
        hist = s.get(f"{BASE}/api/testchat/{emp['id']}/sessions/{sid}/messages",
                     headers=h, timeout=20).json()
        said = [m for m in hist["items"] if m["role"] in ("visitor", "ai")]
        roles = [m["role"] for m in said]
        ai_msgs = [m["content"] for m in said if m["role"] == "ai"]
        print(f"        历史条数：{len(hist['items'])}（对话双方 {len(said)} 条）")
        check(roles == ["visitor", "ai"] * 4, "四轮问答的角色序列完整", str(roles))
        check(all(ai_msgs), "每条 AI 回复都落库且非空", f"空的 AI 消息：{ai_msgs}")
        check(all(m.get("created_at") for m in said), "历史消息带时间戳")

        # ------------------------------------------------------------ 隔离
        print("\n[7] 隔离：测试不得影响线上")
        listed = s.get(f"{BASE}/api/sessions?limit=200", headers=h, timeout=30).json()["items"]
        check(all(x["id"] != sid for x in listed), "测试会话不出现在坐席会话列表")

        overview_after = s.get(f"{BASE}/api/insights/overview?days=7", headers=h, timeout=30).json()
        if overview_before.get("empty"):
            check(overview_after.get("empty"), "概览仍为空态（测试未把空态变成有数据）")
        else:
            b = overview_before["metrics"]["sessions_total"]
            a = overview_after["metrics"]["sessions_total"]
            check(a == b, f"概览会话总数未变（{b} → {a}）", "测试会话被算进了线上统计")

    finally:
        # ------------------------------------------------------------ 清理
        print("\n[8] 清理测试会话")
        d = s.delete(f"{BASE}/api/testchat/{emp['id']}/sessions/{sid}", headers=h, timeout=20)
        check(d.status_code == 200, "删除测试会话", d.text[:200])
        gone = s.get(f"{BASE}/api/testchat/{emp['id']}/sessions/{sid}/messages",
                     headers=h, timeout=20)
        check(gone.status_code == 404, "删除后历史不可读")
        left = s.get(f"{BASE}/api/testchat/{emp['id']}/sessions", headers=h, timeout=20).json()["items"]
        check(all(x["id"] != sid for x in left), "删除后不在测试会话列表里")

    print("\n" + "=" * 62)
    print(f"通过 {PASS}　失败 {FAIL}")
    for n in NOTES:
        print(f"  · {n}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
