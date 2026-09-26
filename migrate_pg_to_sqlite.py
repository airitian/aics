"""一次性迁移：Neon PostgreSQL -> 本地 SQLite（退回本地方案）。

用法：python migrate_pg_to_sqlite.py
- 源：.env 里的 DATABASE_URL（Neon，含 search_path=aics）
- 目标：sqlite:///./aics.db（旧文件已先备份为 aics.db.bak-pre-sqlite-return）
- 全量覆盖式：目标库表须为空（脚本会先清掉旧 sqlite 文件重建）。
"""
from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text

from app.config import settings
from app.database import Base
from app import models  # noqa: F401 确保全部表注册进 metadata

if os.path.exists("aics.db"):
    os.remove("aics.db")
# 清掉 WAL 残留
for suffix in ("-shm", "-wal"):
    if os.path.exists("aics.db" + suffix):
        os.remove("aics.db" + suffix)

SRC_URL = settings.database_url
src_engine = create_engine(SRC_URL, connect_args={"prepare_threshold": 0})
dst_engine = create_engine("sqlite:///./aics.db")

Base.metadata.create_all(bind=dst_engine)


def main() -> int:
    tables = Base.metadata.sorted_tables
    total_rows = 0
    with src_engine.connect() as src, dst_engine.begin() as dst:
        for table in tables:
            n = dst.execute(text(f'SELECT COUNT(*) FROM "{table.name}"')).scalar_one()
            if n:
                print(f"ABORT: 目标表 {table.name} 已有 {n} 行")
                return 1

        for table in tables:
            rows = [dict(r._mapping) for r in src.execute(text(f'SELECT * FROM "{table.name}"'))]
            if not rows:
                print(f"  {table.name}: 0 行")
                continue
            chunk = 500
            for i in range(0, len(rows), chunk):
                dst.execute(table.insert(), rows[i : i + chunk])
            total_rows += len(rows)
            print(f"  {table.name}: {len(rows)} 行")

    print(f"DONE: 共迁移 {total_rows} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
