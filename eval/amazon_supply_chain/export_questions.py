"""把 questions.json 导出成人读的 Markdown 题库和可导入 Excel 的 CSV。

用法：
    python export_questions.py
    python export_questions.py --out-dir .

产物：
    amazon供应链标准手册_测试题库.md      —— 人工测试时对着念的题库
    amazon供应链标准手册_测试题库.csv     —— 可导入 Excel / 测试管理工具
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHECK_LABEL = {
    "contains_all": "全部关键词组命中",
    "contains_any_n": "关键词组命中达标",
    "expect_fallback": "必须兜底/转人工（不得编造）",
    "expect_canned": "必须命中预置话术",
    "manual": "人工判读",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_desc(q: dict) -> str:
    c = q.get("check") or {}
    t = c.get("type", "manual")
    label = CHECK_LABEL.get(t, t)
    if t == "contains_any_n":
        return f"{label}（{c.get('n')}/{len(c.get('groups', []))}）"
    if t == "expect_canned":
        return f"{label}：{c.get('which')}"
    return label


def write_markdown(data: dict, out: Path) -> None:
    meta = data["meta"]
    qs = data["questions"]
    cats: dict[str, list[dict]] = {}
    for q in qs:
        cats.setdefault(q["category"], []).append(q)

    lines: list[str] = []
    lines.append(f"# {meta['title']}")
    lines.append("")
    lines.append(f"- **知识来源**：`{meta['source_doc']}`（{meta['source_doc_note']}）")
    lines.append(f"- **测试对象**：{meta['tenant']} · {meta['knowledge_base']} · AI 员工「{meta['employee']}」")
    lines.append(f"- **题量**：{len(qs)} 题")
    lines.append(f"- **版本**：{meta['version']}（{meta['created_at']}）")
    lines.append("")
    lines.append("> 判分说明：" + meta["scoring_note"])
    lines.append("")

    lines.append("## 分类说明")
    lines.append("")
    lines.append("| 分类 | 说明 | 题数 |")
    lines.append("| --- | --- | --- |")
    for code, desc in meta["category_legend"].items():
        n = len(cats.get(code, []))
        if n:
            lines.append(f"| {code} | {desc} | {n} |")
    lines.append("")

    for code in sorted(cats):
        items = cats[code]
        lines.append(f"## {code} 类 · {meta['category_legend'].get(code, '')}")
        lines.append("")
        for q in items:
            lines.append(f"### {q['id']}　难度：{q['difficulty']}")
            lines.append("")
            lines.append(f"**问**：{q['question']}")
            lines.append("")
            lines.append(f"**标准答案**：{q['expected']}")
            lines.append("")
            lines.append(f"**出处**：{q['source']}　|　**判分**：{check_desc(q)}")
            if q.get("note"):
                lines.append("")
                lines.append(f"**判分备注**：{q['note']}")
            lines.append("")
            lines.append("---")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def write_csv(data: dict, out: Path) -> None:
    qs = data["questions"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["编号", "分类", "难度", "测试问题", "标准答案", "出处", "判分方式", "判分备注"])
        for q in qs:
            w.writerow([
                q["id"],
                q["category"],
                q["difficulty"],
                q["question"],
                q["expected"],
                q["source"],
                check_desc(q),
                q.get("note", ""),
            ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    data = load(Path(args.questions))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = out_dir / "amazon供应链标准手册_测试题库.md"
    csv_path = out_dir / "amazon供应链标准手册_测试题库.csv"
    write_markdown(data, md)
    write_csv(data, csv_path)

    qs = data["questions"]
    counts: dict[str, int] = {}
    for q in qs:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    print(f"题量 {len(qs)}：" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"已写出：\n  {md}\n  {csv_path}")


if __name__ == "__main__":
    main()
