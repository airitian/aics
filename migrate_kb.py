# -*- coding: utf-8 -*-
"""一次性迁移脚本：把默认知识库里的 3 个文件拆到各自的新知识库。

步骤：登录 → 建库（已存在则跳过）→ 上传文件 → 删除默认库中的重复文档。
结果写入 _migrate_report.txt（UTF-8），不走控制台避免编码问题。
"""
from __future__ import annotations

import sys

import requests

BASE = "http://127.0.0.1:8000"
REPORT = r"C:\Users\Administrator\WorkBuddy\2026-09-14-17-02-15\_migrate_report.txt"
DEFAULT_KB = "a647f2d9bccc419da0d7e57a7251fa42"

# (新库名, 源文件绝对路径)
PLAN = [
    ("Amazon 供应链标准", r"C:\Users\Administrator\Desktop\amazon-supplier-manual-simplified-chinese.pdf"),
    ("需求文档", r"C:\Users\Administrator\Desktop\需求文档.docx"),
    ("地面清洁机器人说明书", r"C:\Users\Administrator\Downloads\20240429100956347.pdf"),
]
# 默认库里要删的重复文件（客服常见问题.txt 保留）
REMOVE_FROM_DEFAULT = {
    "amazon-supplier-manual-simplified-chinese.pdf",
    "需求文档.docx",
    "20240429100956347.pdf",
}

lines: list[str] = []


def log(msg: str) -> None:
    lines.append(msg)
    print(msg, file=sys.stderr)


def main() -> None:
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"email": "star@demo.local", "password": "demo12345"}, timeout=30)
    r.raise_for_status()
    token = r.json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    log("login ok")

    # 1. 建库
    kb_ids: dict[str, str] = {}
    for name, _ in PLAN:
        rr = s.post(f"{BASE}/api/knowledge-bases", json={"name": name}, timeout=30)
        if rr.status_code == 200:
            kb_ids[name] = rr.json()["id"]
            log(f"kb created: {name} -> {kb_ids[name]}")
        elif rr.status_code == 409:
            # 已存在：从列表里找 id
            items = s.get(f"{BASE}/api/knowledge-bases", timeout=30).json()["items"]
            kb_ids[name] = next(k["id"] for k in items if k["name"] == name)
            log(f"kb exists: {name} -> {kb_ids[name]}")
        else:
            raise SystemExit(f"create kb failed {rr.status_code}: {rr.text}")

    # 2. 上传文件
    for name, path in PLAN:
        fn = path.rsplit("\\", 1)[-1]
        with open(path, "rb") as f:
            rr = s.post(
                f"{BASE}/api/knowledge-bases/{kb_ids[name]}/documents",
                files={"files": (fn, f)},
                timeout=600,
            )
        ok = rr.status_code == 200
        log(f"upload {'ok' if ok else 'FAIL'}: {fn} -> {name} ({rr.status_code}) {'' if ok else rr.text[:200]}")

    # 3. 删除默认库中的重复文档
    docs = s.get(f"{BASE}/api/knowledge-bases/{DEFAULT_KB}/documents", timeout=30).json()["items"]
    for d in docs:
        fname = d.get("filename") or d.get("name") or ""
        if fname in REMOVE_FROM_DEFAULT:
            dd = s.delete(f"{BASE}/api/documents/{d['id']}", timeout=120)
            log(f"remove from default: {fname} ({dd.status_code})")
        else:
            log(f"keep in default: {fname}")

    # 4. 最终分布
    items = s.get(f"{BASE}/api/knowledge-bases", timeout=30).json()["items"]
    log("---- final distribution ----")
    for k in items:
        log(f"{k['name']}: {k['doc_count']} docs")


if __name__ == "__main__":
    try:
        main()
        lines.insert(0, "RESULT=OK")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"RESULT=FAIL {exc!r}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
