"""一次性迁移：Neon PostgreSQL -> 本地 PostgreSQL。

用法：python migrate_pg_to_pg.py <本地连接串>
示例：python migrate_pg_to_pg.py "postgresql+psycopg://aics:aics@127.0.0.1:5432/aics"
- 源：.env 里的 DATABASE_URL（Neon）
- 目标：命令行传入的本地连接串
- 先 create_all 建全表，再按外键依赖顺序逐表搬运（全量覆盖式，目标须为空库）。
"""
from __future__ import annotations

import sys

from sqlalchemy import create_engine, text

from app.database import Base
from app import models  # noqa: F401 确保全部表注册进 metadata

if len(sys.argv) < 2:
    print("用法: python migrate_pg_to_pg.py <本地目标连接串>")
    sys.exit(2)

SRC_URL = None  # 从 .env 读
from app.config import settings  # noqa: E402

SRC_URL = settings.database_url
DST_URL = sys.argv[1]

src_engine = create_engine(SRC_URL, connect_args={"prepare_threshold": 0})
dst_engine = create_engine(DST_URL)


def main() -> int:
    with dst_engine.begin() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "aics"'))

    # 目标走 aics schema
    dst_engine_local = create_engine(DST_URL, connect_args={"options": "-csearch_path=aics"})
    Base.metadata.create_all(bind=dst_engine_local)

    tables = Base.metadata.sorted_tables
    total_rows = 0
    with src_engine.connect() as src, dst_engine_local.begin() as dst:
        for table in tables:
            n = dst.execute(text(f'SELECT COUNT(*) FROM "{table.name}"')).scalar_one()
            if n:
                print(f"ABORT: 目标表 {table.name} 已有 {n} 行，为防重复搬运请先清库")
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
