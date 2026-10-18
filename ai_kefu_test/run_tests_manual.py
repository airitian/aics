# -*- coding: utf-8 -*-
"""执行 47 条说明书场景测试用例（走 AICS testchat 真实链路 + dev 档 LLM）并判分。

- 每条用例独立会话（多轮用例在同一会话内按轮次顺序发送）；
- 全部请求带 X-LLM-Profile: dev → 走旧中转 key（开发测试档，省官方额度）；
- 判分：直接调 dev 档模型做 LLM-as-judge，对照 expected / judge_criteria 给
  pass / partial / fail + 差异说明；
- 产物：results_manual.json（全量明细）+ 控制台差异摘要。
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 本目录：随仓库走，clone 到任何机器都能跑（不再硬编码本机路径）
HERE = os.path.dirname(os.path.abspath(__file__))

PROXIES = {"http": None, "https": None}
DEV_HEADER = {"X-LLM-Profile": "dev"}


def _load_local_env() -> None:
    """把同目录 .env 的键值灌进 os.environ（不覆盖已有环境变量）。

    不引 python-dotenv：就几行的事。优先顺序：进程环境变量 > .env > 内置默认。
    """
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_local_env()

# 被测服务地址与员工。本地 dev 默认 8000（run.py 默认端口；.env 未配 PORT），
# 生产是 8001——用 AICS_BASE 覆盖。
BASE = os.environ.get("AICS_BASE", "http://127.0.0.1:8000")
EMP_ID = os.environ.get("AICS_EMP_ID", "34e43866643e47e1a5e097386ce1a5ef")

# 裁判模型配置。
# 2026-10-18 起裁判与被测**异构**：被测是 deepseek 系（dev 档 deepseek-v4.1-flash
# 走 pateway），裁判换成 pateway 上的 GPT 系 gpt-6-sol——同模型当裁判会
# 自我偏好 + 同源错误互盲，基线分数虚高（详见评测缺口报告 P0）。
# ⚠️ 换裁判 = 换判分标准：历史基线分数全部作废，必须重建基线后再对比。
# 默认 URL/key 仅为兼容旧用法；请用同目录 .env 或环境变量显式覆盖。
JUDGE_URL = os.environ.get("JUDGE_URL", "https://api.pateway.ai/v1/chat/completions")
JUDGE_KEY = os.environ.get("JUDGE_KEY", "sk-ptw-132WnNhtW9LvAwOT6A98bGBYFcVB3qaUzAb9e")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-6-sol")


def new_session() -> requests.Session:
    s = requests.Session()
    s.proxies = PROXIES
    r = s.post(f"{BASE}/api/auth/auto", timeout=15)
    r.raise_for_status()
    s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return s


def split_turns(raw: str) -> list[str]:
    """『第一轮：xxx｜第二轮：yyy』→ [xxx, yyy]；无轮次标记就单轮。"""
    if "第一轮" not in raw:
        return [raw.strip()]
    parts = re.split(r"[｜|]", raw)
    turns = []
    for p in parts:
        p = re.sub(r"^第[一二三四五]轮[：:]", "", p.strip()).strip()
        if p:
            turns.append(p)
    return turns or [raw.strip()]


def run_case(s: requests.Session, case: dict) -> dict:
    turns = split_turns(case["input"])
    out = {**case, "turns": turns, "replies": [], "error": "", "latency_ms": 0}
    started = time.perf_counter()
    try:
        r = s.post(f"{BASE}/api/testchat/{EMP_ID}/sessions", timeout=15)
        r.raise_for_status()
        sid = r.json()["id"]
        out["session_id"] = sid
        transcript = []
        for i, msg in enumerate(turns, 1):
            r = s.post(
                f"{BASE}/api/testchat/{EMP_ID}/sessions/{sid}/messages",
                json={"message": msg},
                headers=DEV_HEADER,
                timeout=180,
            )
            if r.status_code != 200:
                out["error"] = f"turn{i} http={r.status_code}: {r.text[:200]}"
                break
            d = r.json()
            out["replies"].append(d.get("reply") or "")
            out["latency_ms"] += int(d.get("latency_ms") or 0)
            out.setdefault("top_score", 0.0)
            out["top_score"] = max(out["top_score"], float(d.get("top_score") or 0))
            out["degraded"] = bool(d.get("degraded"))
            hits = d.get("hits") or []
            if hits:
                out.setdefault("hit_files", set())
                out["hit_files"].add(hits[0].get("filename") or "")
        if isinstance(out.get("hit_files"), set):
            out["hit_files"] = sorted(out["hit_files"])
        # 测试会话用完即删：不清理会一直堆在库里（每轮 47 个）
        try:
            s.delete(f"{BASE}/api/testchat/{EMP_ID}/sessions/{sid}", timeout=15)
        except Exception:  # noqa: BLE001 - 清理失败不影响结果
            pass
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_ms"] = int((time.perf_counter() - started) * 1000)
    return out


JUDGE_SYSTEM = (
    "你是 AI 客服评测的严格判分员。根据【标准答案】【判分标准】对【AI 实际回复】判分。\n"
    "规则：\n"
    "- pass：事实与标准答案一致、满足判分标准；措辞不同没关系。\n"
    "- partial：方向对但有遗漏/含糊（如多原因排查只答了一半）。\n"
    "- fail：事实错误、编造数字/条款、答非所问，或该拒答的没拒答。\n"
    "- 判分标准里的每一条硬性要求（数字精确、不编造、需澄清等）都必须核对。\n"
    "- **额外补充不扣分**：AI 回复中包含标准答案之外的有益信息（附加提醒、"
    "安全提示、相关建议）不算差异，不要因此降级；只核对标准答案要求的信息"
    "是否齐全、事实是否正确。\n"
    "- **语义等价不要求逐字匹配**：条目拆分粒度不同（如一条原因拆成两条更细）、"
    "表述顺序不同、措辞不同，只要语义覆盖标准答案的全部要点即算满足。\n"
    "- **合理路径等同**：排查/引导类问题若 AI 走了与标准答案不同但同样合理的"
    "路径（如追问现象细节 vs 指定排查步骤），且覆盖了该场景的关键动作，不算遗漏。\n"
    "只输出 JSON：{\"verdict\":\"pass|partial|fail\",\"diff\":\"与标准答案的差异，无差异填空字符串\",\"reason\":\"一句话理由\"}"
)


def _adapt_body_for_model(model: str, body: dict) -> dict:
    """按模型调整请求体，兼容 OpenAI 新旧两套参数规范。

    实测 pateway 上 gpt-6-sol 的两处硬差异（都直接 400）：
    1. 不认 `max_tokens`，只认 `max_completion_tokens`；
    2. 不认 `temperature=0.0`，只支持默认值 1（o 系推理模型的通例）。
    deepseek/glm/kimi 等老模型相反：只认 max_tokens，temperature=0 有效。
    判分 token 上限必须显式传——思维链模型不设上限会把预算烧光、
    正文截断（历史教训：判分上限 500→2000）。
    """
    body = dict(body)
    if model.startswith(("gpt-6", "gpt-5.6", "gpt-5.5", "o1", "o3", "o4")):
        body.pop("temperature", None)
        body.pop("max_tokens", None)
        body["max_completion_tokens"] = 2000
    else:
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = 2000
        body["temperature"] = 0.0
    return body


def judge(case: dict) -> dict:
    transcript = ""
    turns = case.get("turns") or [case["input"]]
    replies = case.get("replies") or []
    for i, t in enumerate(turns, 1):
        transcript += f"\n【用户第{i}轮】{t}\n【AI 第{i}轮】{(replies[i-1] if i-1 < len(replies) else '(无回复)')}"
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"【问题类别】{case['category']} / {case['sub_type']}\n"
                    f"【标准答案】{case['expected']}\n"
                    f"【判分标准】{case['judge_criteria']}\n"
                    f"【AI 实际回复】{transcript}"
                ),
            },
        ],
        "stream": False,
    }
    body = _adapt_body_for_model(JUDGE_MODEL, body)
    # 中转 key 有速率限制：429 必须退避重试，否则判分全被打成 fail（第一轮实测教训）
    last_exc: Exception | None = None
    param_switched = False
    for attempt in range(5):
        try:
            r = requests.post(
                JUDGE_URL,
                json=body,
                headers={"Authorization": f"Bearer {JUDGE_KEY}"},
                proxies=PROXIES,
                timeout=90,
            )
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    [judge 429] {case['id']} 等待 {wait}s 重试", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 400 and not param_switched:
                # 参数规范猜错（模型名前缀规则没覆盖到的新模型）：换另一套规范重试
                err = r.text
                if "max_completion_tokens" in err or "temperature" in err:
                    param_switched = True
                    if "max_completion_tokens" in body:
                        body.pop("max_completion_tokens")
                        body["max_tokens"] = 2000
                        body["temperature"] = 0.0
                    else:
                        body.pop("temperature", None)
                        body.pop("max_tokens", None)
                        body["max_completion_tokens"] = 2000
                    print(f"    [judge 参数切换] {case['id']} 切换参数规范重试", flush=True)
                    continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return {"verdict": "fail", "diff": "判分输出不可解析", "reason": text[:200]}
            return json.loads(m.group(0))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(3)
    return {"verdict": "fail", "diff": f"判分调用失败: {last_exc}", "reason": ""}


def main() -> None:
    tc = json.load(open(os.path.join(HERE, "test_cases_manual.json"), encoding="utf-8"))
    cases = tc["cases"]
    s = new_session()
    # 串行执行：中转 key 并发 4 就触发 429 限流（第一轮实测），
    # 聊天侧一旦 429 会被 llm.py 当作「模型不可用」转人工，结果全废。
    print(f"[run] {len(cases)} cases, sequential ...", flush=True)

    results: dict[str, dict] = {}
    for idx, c in enumerate(cases, 1):
        try:
            results[c["id"]] = run_case(s, c)
        except Exception as exc:  # noqa: BLE001
            results[c["id"]] = {"id": c["id"], "error": str(exc), "replies": []}
        ok = not results[c["id"]].get("error")
        print(f"  [{idx}/{len(cases)}] {c['id']} ok={ok}", flush=True)
        time.sleep(1.5)  # 给中转限流留出余量

    print("[judge] ...", flush=True)
    for cid in results:
        results[cid]["judge"] = judge(results[cid])

    ordered = [results[c["id"]] for c in cases if c["id"] in results]
    for r in ordered:
        hf = r.pop("hit_files", None)
        if hf is not None and not isinstance(hf, list):
            r["hit_files"] = sorted(hf)
    json.dump(
        {"meta": {"total": len(ordered), "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "llm_profile": "dev", "employee": EMP_ID},
         "results": ordered},
        open(os.path.join(HERE, "results_manual.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1,
    )

    # 控制台摘要：差异明细
    from collections import Counter
    cnt = Counter(r["judge"]["verdict"] for r in ordered if r.get("judge"))
    print("\n===== 汇总 =====")
    print(dict(cnt))
    print("\n===== 差异明细（partial / fail）=====")
    for r in ordered:
        v = (r.get("judge") or {}).get("verdict")
        if v in ("partial", "fail"):
            print(f"\n{r['id']} [{v}] {r['category']}/{r['sub_type']}")
            print(f"  问: {r['input'][:80]}")
            print(f"  差异: {(r['judge'] or {}).get('diff','')[:200]}")
            print(f"  理由: {(r['judge'] or {}).get('reason','')[:150]}")
            if r.get("error"):
                print(f"  错误: {r['error'][:150]}")
            elif r.get("replies"):
                print(f"  答: {r['replies'][0][:150]}")


if __name__ == "__main__":
    main()
