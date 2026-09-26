"""Amazon 供应链标准手册 · AI 客服准确率自动评测。

走的是系统内置的「问答测试（Playground）」链路：不影响线上会话、不产生工单，
所以可以放心反复跑。

用法：
    python run_accuracy_eval.py                    # 跑全量题库，出报告
    python run_accuracy_eval.py --limit 10         # 只跑前 10 题
    python run_accuracy_eval.py --category H       # 只跑某一类（可逗号分隔）
    python run_accuracy_eval.py --base http://127.0.0.1:8000

退出码：0 = 全部通过；1 = 有未通过项。方便挂到 CI 或定时任务上。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 系统预置话术（与 app/prompting.py 保持一致）
# --------------------------------------------------------------------------- #
CANNED = {
    "no_answer": "我这边暂时没有查到相关资料",
    "handoff": "好的，我这就为您转接人工客服",
    "handoff_queued": "已为您记录，当前人工客服不在线",
    "degraded": "抱歉，我暂时无法处理您的消息",
    "clarify": "抱歉，我没太理解您的意思",
    "ad": "不好意思，这里只受理本业务的咨询",
    "abuse": "抱歉给您带来不好的体验",
}
FALLBACK_PREFIXES = (CANNED["no_answer"], CANNED["handoff"], CANNED["handoff_queued"], CANNED["degraded"])
CANNED_BY_WHICH = {
    "handoff": (CANNED["handoff"], CANNED["handoff_queued"], CANNED["degraded"]),
    "clarify": (CANNED["clarify"],),
    "ad": (CANNED["ad"],),
    "abuse": (CANNED["abuse"],),
}

# 「查不到」的自然表述 —— 模型自己说没有，也算合格的兜底。
#
# 必须用**正则**而不是字面子串：模型不会照抄系统的兜底话术，而是把它嵌进自己的句子里，
# 典型如「我们这里暂时没有亚马逊供应商账号和密码的**相关资料**」——
# 字面匹配「没有相关资料」会漏掉这一句，把正确的「承认查不到」判成「编造」。
# 这是最容易冤枉模型的一类误判，宁可放宽也不要误伤（真正的编造由 must_not 数字断言兜底）。
NO_INFO_PATTERNS = (
    r"没有[^。！？；\n]{0,18}(资料|信息|内容|数据|说明|记录|文件)",
    r"没有(查到|找到|收录|提及|涉及|规定|明确|给出|写明|说明|列明|披露|提供具体)",
    r"未(查到|找到|收录|提及|涉及|规定|明确|给出|写明|说明|列明|披露)",
    r"查不到|找不到|不清楚|无法提供|无法确认|不能确认|不掌握",
    r"暂(时)?(没|未)有",
    r"没有这方面",
    r"无相关(资料|信息|内容)",
)


def is_fallback(reply: str, handoff: bool) -> bool:
    if handoff:
        return True
    if not reply:
        return True
    if reply.startswith(FALLBACK_PREFIXES):
        return True
    return any(re.search(p, reply) for p in NO_INFO_PATTERNS)


def _patterns(raw) -> list[str]:
    """把 must_not / review_if 归一化成扁平的字符串列表。

    题库里手写 JSON 很容易把 ["A", "B"] 误写成 [["A", "B"]]（写成关键词组的样子），
    这种错误只有在跑了几十题之后才会以 TypeError 的形式炸出来。这里直接容错，
    同时 validate_questions() 会在开跑前把它报出来。
    """
    out: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item if isinstance(x, str))
    return out


def _any_match(reply: str, patterns: list[str]) -> str:
    for p in patterns:
        try:
            if re.search(p, reply, re.IGNORECASE):
                return p
        except re.error:
            if p.lower() in reply.lower():
                return p
    return ""


def validate_questions(questions: list[dict]) -> list[str]:
    """开跑前结构自检，把问题一次性列出来而不是跑到一半才炸。"""
    problems: list[str] = []
    seen: set[str] = set()
    valid_types = {"contains_all", "contains_any_n", "expect_fallback", "expect_canned", "manual"}
    for q in questions:
        qid = q.get("id", "?")
        if qid in seen:
            problems.append(f"{qid}：编号重复")
        seen.add(qid)
        for field in ("question", "expected", "category"):
            if not q.get(field):
                problems.append(f"{qid}：缺少 {field}")
        check = q.get("check") or {}
        ctype = check.get("type")
        if ctype not in valid_types:
            problems.append(f"{qid}：未知判分类型 {ctype!r}")
        if ctype in ("contains_all", "contains_any_n"):
            groups = check.get("groups") or []
            if not groups:
                problems.append(f"{qid}：{ctype} 缺少 groups")
            for g in groups:
                if not isinstance(g, list) or not g or not all(isinstance(x, str) for x in g):
                    problems.append(f"{qid}：groups 里有非法元素 {g!r}")
            if ctype == "contains_any_n":
                n = check.get("n")
                if not isinstance(n, int) or not (1 <= n <= len(groups)):
                    problems.append(f"{qid}：n={n!r} 越界（应为 1..{len(groups)}）")
        if ctype == "expect_canned" and check.get("which") not in CANNED_BY_WHICH:
            problems.append(f"{qid}：expect_canned.which={check.get('which')!r} 不认识")
        for field in ("must_not", "review_if"):
            raw = q.get(field)
            if raw is None:
                continue
            if not isinstance(raw, list):
                problems.append(f"{qid}：{field} 不是列表")
                continue
            if any(isinstance(x, (list, tuple)) for x in raw):
                problems.append(f"{qid}：{field} 出现嵌套列表（应为扁平字符串数组，已自动纠正）")
    return problems


def evaluate(q: dict, item: dict) -> dict:
    """对单题判分。返回 pass / reasons / review / observed。"""
    reply = (item.get("answer") or "").strip()
    handoff = bool(item.get("handoff"))
    err = item.get("error") or ""
    check = q.get("check") or {"type": "manual"}
    ctype = check.get("type", "manual")

    reasons: list[str] = []
    review: list[str] = []
    passed: bool | None

    if err or item.get("ok") is False:
        passed = False
        reasons.append(f"调用失败：{err or 'ok=false'}")

    elif ctype == "contains_all":
        missing = [
            "/".join(g) for g in check.get("groups", [])
            if not any(t.lower() in reply.lower() for t in g)
        ]
        passed = not missing
        if missing:
            reasons.append("缺少要点：" + "；".join(missing))

    elif ctype == "contains_any_n":
        groups = check.get("groups", [])
        need = check.get("n", len(groups))
        hit = [g for g in groups if any(t.lower() in reply.lower() for t in g)]
        passed = len(hit) >= need
        if not passed:
            reasons.append(f"要点命中不足：命中 {len(hit)}/{need}（命中：{'、'.join(g[0] for g in hit) or '无'}）")

    elif ctype == "expect_fallback":
        passed = is_fallback(reply, handoff)
        if not passed:
            reasons.append("本该承认查不到，却给出了具体答案（疑似编造）")

    elif ctype == "expect_canned":
        which = check.get("which", "")
        prefixes = CANNED_BY_WHICH.get(which, ())
        passed = (handoff and which == "handoff") or reply.startswith(prefixes)
        if not passed:
            reasons.append(f"未命中预置话术：{which}")

    else:  # manual
        passed = None

    # 硬失败项
    for p in _patterns(q.get("must_not")):
        hit = _any_match(reply, [p])
        if hit:
            passed = False
            reasons.append(f"命中禁止内容：/{hit}/")

    # 软标记：只提示人工复核，不判失败
    for p in _patterns(q.get("review_if")):
        hit = _any_match(reply, [p])
        if hit:
            review.append(f"命中复核模式：/{hit}/")

    hits = item.get("hits") or []
    return {
        "id": q["id"],
        "category": q["category"],
        "difficulty": q.get("difficulty", ""),
        "question": q["question"],
        "expected": q["expected"],
        "source": q.get("source", ""),
        "reply": reply,
        "passed": passed,
        "reasons": reasons,
        "review": review,
        "handoff": handoff,
        "degraded": bool(item.get("degraded")),
        "degrade_reason": item.get("degrade_reason", ""),
        "top_score": round(float(item.get("top_score") or 0.0), 4),
        "hit_count": len(hits),
        "top_file": (hits[0].get("filename") if hits and isinstance(hits[0], dict) else ""),
        "latency_ms": item.get("latency_ms", 0),
        "tokens": item.get("tokens", 0),
    }


def run(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions: list[dict] = data["questions"]
    if args.category:
        wanted = {c.strip().upper() for c in args.category.split(",") if c.strip()}
        questions = [q for q in questions if q["category"].upper() in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("没有可执行的题目（检查 --category / --limit）")
        return 1

    problems = validate_questions(questions)
    if problems:
        print(f"⚠ 题库自检发现 {len(problems)} 处问题：")
        for p in problems:
            print(f"  - {p}")

    base = args.base.rstrip("/")
    print(f"服务地址：{base}")
    print(f"待测题量：{len(questions)}")

    with httpx.Client(base_url=base, timeout=args.timeout) as c:
        try:
            health = c.get("/healthz").json()
        except Exception as exc:  # noqa: BLE001
            print(f"服务不可达：{exc}")
            return 1
        if health.get("degraded"):
            print(f"⚠ 服务处于降级状态：{json.dumps(health, ensure_ascii=False)[:300]}")

        r = c.post("/api/auth/login", json={"email": args.email, "password": args.password})
        if r.status_code != 200:
            print(f"登录失败 {r.status_code}：{r.text[:200]}")
            return 1
        token = r.json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        emp_rows = c.get("/api/employees", headers=h).json().get("items", [])
        if not emp_rows:
            print("该租户下没有 AI 员工")
            return 1
        emp = None
        if args.employee:
            emp = next((e for e in emp_rows if e["name"] == args.employee or e["id"] == args.employee), None)
        if emp is None:
            emp = next((e for e in emp_rows if e.get("is_default")), emp_rows[0])
        print(f"AI 员工：{emp['name']}（{emp['id']}）")

        batch_max = args.batch
        results: list[dict] = []

        for i in range(0, len(questions), batch_max):
            batch = questions[i : i + batch_max]
            chunk = [q["question"] for q in batch]
            resp = c.post(
                f"/api/playground/{emp['id']}/batch",
                headers=h,
                json={"questions": chunk},
                timeout=args.timeout,
            )
            if resp.status_code != 200:
                print(f"批次 {i // batch_max + 1} 失败 {resp.status_code}：{resp.text[:300]}")
                return 1
            body = resp.json()
            items = body.get("items", [])
            if len(items) != len(chunk):
                print(f"⚠ 返回条数 {len(items)} 与提交条数 {len(chunk)} 不一致")
            if body.get("rejected_count"):
                print(f"⚠ 服务端拒绝了 {body['rejected_count']} 条（超过单批上限 {body.get('max_batch')}）")
            for q, item in zip(batch, items):
                results.append(evaluate(q, item))
            print(f"  已跑 {min(i + batch_max, len(questions))}/{len(questions)}")

    # ------------------------------------------------------------------ 统计 + 报告
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return emit_report(
        meta=data["meta"],
        results=results,
        run_meta={
            "at": datetime.now().isoformat(timespec="seconds"),
            "base": base,
            "employee": emp["name"],
            "employee_id": emp["id"],
            "healthz": health,
            "rescored_from": "",
        },
        out_dir=out_dir,
    )


def emit_report(*, meta: dict, results: list[dict], run_meta: dict, out_dir: Path) -> int:
    """汇总统计并写出 Markdown 报告 + 原始 JSON。返回退出码。"""
    graded = [r for r in results if r["passed"] is not None]
    ok = [r for r in graded if r["passed"]]
    fail = [r for r in graded if not r["passed"]]
    need_review = [r for r in results if r["review"]]
    total = len(graded)
    accuracy = (len(ok) / total * 100) if total else 0.0

    cats: dict[str, dict] = {}
    for r in graded:
        c = cats.setdefault(r["category"], {"total": 0, "ok": 0})
        c["total"] += 1
        c["ok"] += 1 if r["passed"] else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"eval_result_{stamp}.json"
    md_path = out_dir / f"eval_report_{stamp}.md"

    payload = {
        "meta": meta,
        "run": {
            **run_meta,
            "questions": len(results),
            "graded": total,
            "passed": len(ok),
            "failed": len(fail),
            "accuracy": round(accuracy, 1),
        },
        "by_category": {k: {**v, "rate": round(v["ok"] / v["total"] * 100, 1)} for k, v in cats.items()},
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    health = run_meta.get("healthz") or {}
    lines: list[str] = []
    lines.append("# AI 客服准确率评测报告")
    lines.append("")
    lines.append(f"- **题库**：{meta['title']} v{meta['version']}")
    lines.append(f"- **知识来源**：`{meta['source_doc']}`")
    lines.append(f"- **被测对象**：{meta['tenant']} · AI 员工「{run_meta.get('employee', '')}」")
    lines.append(f"- **执行时间**：{run_meta.get('at', '')}")
    lines.append(f"- **模型**：{health.get('llm', {}).get('model', '未知')}　"
                 f"**向量模型**：{health.get('embed', {}).get('model', '未知')}　"
                 f"**向量库**：{health.get('vector', {}).get('backend', '未知')}")
    if run_meta.get("rescored_from"):
        lines.append(f"- **来源**：由既有结果重新判分（未重新提问）：`{run_meta['rescored_from']}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 执行题数 | {len(results)} |")
    lines.append(f"| 参与判分 | {total} |")
    lines.append(f"| 通过 | {len(ok)} |")
    lines.append(f"| 未通过 | **{len(fail)}** |")
    lines.append(f"| 准确率 | **{accuracy:.1f}%** |")
    lines.append(f"| 需人工复核 | {len(need_review)} |")
    lines.append("")
    lines.append("## 分类表现")
    lines.append("")
    lines.append("| 分类 | 说明 | 通过/总数 | 准确率 |")
    lines.append("| --- | --- | --- | --- |")
    for code in sorted(cats):
        c = cats[code]
        desc = meta["category_legend"].get(code, "")
        lines.append(f"| {code} | {desc} | {c['ok']}/{c['total']} | {c['ok'] / c['total'] * 100:.1f}% |")
    lines.append("")

    if fail:
        lines.append(f"## 未通过明细（{len(fail)} 题）")
        lines.append("")
        for r in fail:
            lines.append(f"### {r['id']}　{r['question']}")
            lines.append("")
            lines.append(f"- **标准答案**：{r['expected']}")
            lines.append(f"- **AI 实际回答**：{r['reply'] or '（空）'}")
            lines.append(f"- **未通过原因**：{'；'.join(r['reasons'])}")
            lines.append(f"- **检索**：命中 {r['hit_count']} 段，最高相似度 {r['top_score']}"
                         + (f"，来源 `{r['top_file']}`" if r['top_file'] else "")
                         + (f"　**转人工**：是" if r['handoff'] else "")
                         + (f"　**降级**：{r['degrade_reason']}" if r['degraded'] else ""))
            lines.append(f"- **出处**：{r['source']}")
            lines.append("")

    if need_review:
        lines.append(f"## 需人工复核（{len(need_review)} 题）")
        lines.append("")
        for r in need_review:
            lines.append(f"- **{r['id']}** {r['question']}")
            lines.append(f"  - 复核原因：{'；'.join(r['review'])}")
            lines.append(f"  - AI 回答：{r['reply'][:200]}")
        lines.append("")

    lines.append("## 全部回放")
    lines.append("")
    lines.append("| 编号 | 分类 | 结果 | 相似度 | 问题 | AI 回答（截断） |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        if r["passed"] is None:
            mark = "—"
        else:
            mark = "通过" if r["passed"] else "**未通过**"
        ans = (r["reply"] or "").replace("\n", " ").replace("|", "／")
        if len(ans) > 80:
            ans = ans[:80] + "…"
        qtext = r["question"].replace("|", "／")
        lines.append(f"| {r['id']} | {r['category']} | {mark} | {r['top_score']} | {qtext} | {ans} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 56)
    print(f"准确率：{accuracy:.1f}%　（通过 {len(ok)} / 判分 {total}，另有 {len(need_review)} 题需复核）")
    for code in sorted(cats):
        c = cats[code]
        print(f"  {code} 类：{c['ok']}/{c['total']}　{c['ok'] / c['total'] * 100:.0f}%")
    if fail:
        print("\n未通过：")
        for r in fail:
            print(f"  {r['id']}　{r['question'][:38]}　→ {'；'.join(r['reasons'])[:60]}")
    print()
    print(f"报告：{md_path}")
    print(f"原始数据：{json_path}")
    return 1 if fail else 0


def rescore(args: argparse.Namespace) -> int:
    """拿既有的结果 JSON 重新判分，不重新提问。

    改判分口径（关键词、兜底词表、软标记）时不用再烧一遍 API ——
    模型的回答是固定的，变的只是我们的尺子。
    """
    src = Path(args.rescore)
    if not src.exists():
        print(f"找不到结果文件：{src}")
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qmap = {q["id"]: q for q in data["questions"]}

    out: list[dict] = []
    missing: list[str] = []
    for r in payload.get("results", []):
        q = qmap.get(r["id"])
        if q is None:
            missing.append(r["id"])
            continue
        # 还原出一个 evaluate() 认识的 item：命中数用同长度占位列表保留下来
        n = int(r.get("hit_count") or 0)
        item = {
            "answer": r.get("reply", ""),
            "handoff": r.get("handoff", False),
            "ok": True,
            "degraded": r.get("degraded", False),
            "degrade_reason": r.get("degrade_reason", ""),
            "top_score": r.get("top_score", 0.0),
            "hits": [{"filename": r.get("top_file", "")}] * n,
            "latency_ms": r.get("latency_ms", 0),
            "tokens": r.get("tokens", 0),
        }
        out.append(evaluate(q, item))

    if missing:
        print(f"⚠ 结果里有 {len(missing)} 题的编号已不在题库中，已跳过：{', '.join(missing)}")

    prev = payload.get("run", {}).get("accuracy")
    return emit_report(
        meta=data["meta"],
        results=out,
        run_meta={
            "at": datetime.now().isoformat(timespec="seconds"),
            "base": payload.get("run", {}).get("base", ""),
            "employee": payload.get("run", {}).get("employee", ""),
            "employee_id": payload.get("run", {}).get("employee_id", ""),
            "healthz": payload.get("run", {}).get("healthz", {}),
            "rescored_from": f"{src.name}（原判分 {prev}%）",
        },
        out_dir=Path(args.out_dir),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Amazon 供应链手册 AI 客服准确率评测")
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--out-dir", default=str(HERE / "reports"))
    ap.add_argument("--base", default=os.environ.get("AICS_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--email", default=os.environ.get("AICS_EVAL_EMAIL", "star@demo.local"))
    ap.add_argument("--password", default=os.environ.get("AICS_EVAL_PASSWORD", "demo12345"))
    ap.add_argument("--employee", default=os.environ.get("AICS_EVAL_EMPLOYEE", ""))
    ap.add_argument("--category", default="", help="只跑指定分类，逗号分隔，如 H 或 A,B")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ap.add_argument("--batch", type=int, default=40, help="每批题数（服务端上限 batch_test_max，默认 50）")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--rescore", default="",
                    help="不重新提问，用当前判分口径对既有结果 JSON 重新判分，如 "
                         "--rescore reports/eval_result_20260923-013532.json")
    args = ap.parse_args()
    if args.rescore:
        sys.exit(rescore(args))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
