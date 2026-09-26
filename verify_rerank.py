#!/usr/bin/env python
"""重排真机验证：走完整检索链路（向量模型 → 向量库 → 重排），确认端到端生效。

和 verify_retrieval.py 的分工：
- verify_retrieval.py：离线统计，横向对比模型，出 Recall@K；
- 本脚本：真机跑几条真实问题，看**排序是否真的被改变**、降级是否可用。

用法
----
    python verify_rerank.py              # 正常验证
    python verify_rerank.py --off        # 临时关闭重排做对照
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.config import settings  # noqa: E402
from app.database import engine  # noqa: E402
from app.models import KnowledgeBase, Tenant  # noqa: E402
from app.rag import retrieve  # noqa: E402

QUESTIONS = [
    "亚马逊对供应商的审计涵盖哪几个类别？",
    "手册是哪一年更新的？",
    "供应商必须给工人提供哪些防护用品？",
    "如果工厂检查发现问题会怎样？",
    "外包生产需要提前报备吗？",
]


async def main(off: bool) -> int:
    if off:
        settings.rerank_provider = "off"
    print(f"重排状态: {settings.rerank_provider}  模型: {settings.rerank_model or '-'}\n")

    with Session(engine) as db:
        tenant = db.execute(select(Tenant)).scalars().first()
        if tenant is None:
            print("库里没有租户，请先启动一次服务完成初始化")
            return 1
        kb = db.execute(
            select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant.id)
        ).scalars().first()
        if kb is None:
            print("库里没有知识库")
            return 1

        for q in QUESTIONS:
            hits, tokens = await retrieve(
                db, tenant_id=tenant.id, kb_ids=[kb.id], query=q
            )
            print(f"Q: {q}")
            if not hits:
                print("   （无命中）\n")
                continue
            for i, h in enumerate(hits, 1):
                print(f"   {i}. [{h.score:.4f}] {h.text[:60].replace(chr(10), ' ')}…")
            print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", action="store_true", help="临时关闭重排做对照")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.off)))
