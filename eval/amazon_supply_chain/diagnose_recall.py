"""「这题为什么答不上来」的排查入口。

对任意问题做一次 top_k=20 的检索，把三件事摆出来：
  1. 前 20 名的相似度分布，以及有多少段过了 MIN_SCORE 门槛
  2. 题目判分用的关键词，各自排在第几名（直接把评分的 needle 当探针，不用另外手工挑词）
  3. 前几名的片段原文

三种结论对应三种修法：

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| 关键词排在第 6 名之后 | top_k 太小，答案被挤出上下文 | 接 rerank，或调大 TOP_K |
| 关键词在前 5 名内、但全部低于 MIN_SCORE | 密集列表稀释了单条相似度 | 改切分策略（按条目/小节） |
| 关键词在前 5 名内、分数也够 | 片段进了上下文但模型没用上 | 调提示词，或加 rerank 提纯 |

用法：
    python diagnose_recall.py A10              # 按题号（关键词从题库自动取）
    python diagnose_recall.py --question "审完要交什么东西？"
    python diagnose_recall.py G03 --top-k 30
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def needles_for(q: dict) -> list[str]:
    """把题目的判分关键词当成定位探针 —— 它本来就是「答案里必须出现的东西」。"""
    check = q.get("check") or {}
    out: list[str] = []
    for group in check.get("groups") or []:
        for token in group:
            # 纯数字或过短的词做探针噪音太大，跳过
            if len(token) >= 3 and not token.isdigit():
                out.append(token)
    return out[:6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qid", nargs="?", default="", help="题号，如 A10")
    ap.add_argument("--question", default="", help="直接给问题文本（与题号二选一）")
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--base", default=os.environ.get("AICS_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--email", default=os.environ.get("AICS_EVAL_EMAIL", "star@demo.local"))
    ap.add_argument("--password", default=os.environ.get("AICS_EVAL_PASSWORD", "demo12345"))
    ap.add_argument("--employee", default=os.environ.get("AICS_EVAL_EMPLOYEE", ""))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--show", type=int, default=3, help="打印前几名的片段原文")
    args = ap.parse_args()

    bank = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    q = None
    if args.qid:
        q = next((x for x in bank["questions"] if x["id"].upper() == args.qid.upper()), None)
        if q is None:
            print(f"题库里没有 {args.qid}")
            return
        question, needles = q["question"], needles_for(q)
    else:
        question, needles = args.question, []
    if not question:
        print("请给题号或 --question")
        return

    gate = float(os.environ.get("MIN_SCORE", "0.45"))
    lines: list[str] = [f"问题：{question}", f"门槛 MIN_SCORE={gate}（如与服务端不一致，请以服务端 .env 为准）", ""]

    with httpx.Client(base_url=args.base, timeout=120) as c:
        tok = c.post("/api/auth/login",
                     json={"email": args.email, "password": args.password}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}
        payload: dict = {"query": question, "top_k": args.top_k}
        if args.employee:
            payload["employee_id"] = args.employee
        hits = c.post("/api/knowledge/search-test", headers=h, json=payload).json().get("items", [])

    scores = [round(x["score"], 4) for x in hits]
    passed = [s for s in scores if s >= gate]
    lines.append(f"前 {len(hits)} 名相似度：{scores}")
    lines.append(f"过门槛的段数：{len(passed)}")
    lines.append("")

    if needles:
        lines.append("关键词（取自本题判分规则）落点：")
        for n in needles:
            rank = [i for i, x in enumerate(hits, 1) if n in (x.get("text") or "")]
            lines.append(f"  「{n}」→ 第 {rank or '（前 ' + str(len(hits)) + ' 名内均无）'} 名")
        lines.append("")

    lines.append("结论参考：")
    top5 = " ".join(x.get("text") or "" for x in hits[:5])
    at_all = " ".join(x.get("text") or "" for x in hits)
    in_top5 = [n for n in needles if n in top5]
    in_any = [n for n in needles if n in at_all]
    if not hits:
        lines.append("  检索完全为空 —— 检查知识库是否已入库、AI 员工是否绑定了该库")
    elif not passed:
        lines.append("  全部低于门槛 → 知识在库里但被 MIN_SCORE 砍掉（密集列表稀释，改切分策略）")
    elif needles and not in_top5 and in_any:
        lines.append(f"  答案块落在第 6 名之后 → top_k 不够，答案被挤出上下文（接 rerank 或调大 TOP_K）")
    elif needles and not in_top5 and not in_any:
        lines.append(f"  关键词在前 {len(hits)} 名内都没出现 → 检索没覆盖到，先查切分是否把内容拆散")
    elif needles and in_top5:
        lines.append("  答案块在前 5 名内 → 若仍答错，问题在提示词或模型归纳，不在检索")
    else:
        lines.append("  本题判分关键词太短、不适合当探针，请直接看下面的片段原文判断")
    lines.append("")

    for i, x in enumerate(hits[: args.show], 1):
        lines.append(f"--- 第 {i} 名（{round(x['score'], 4)}）---")
        lines.append((x.get("text") or "")[:400].replace("\n", " "))
        lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
