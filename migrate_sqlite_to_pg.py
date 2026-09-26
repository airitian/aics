"""一次性迁移：SQLite -> Neon PostgreSQL。

用法：python migrate_sqlite_to_pg.py
- 源：sqlite:///./aics.db（本地现有数据）
- 目标：settings.database_url（.env 里的 Neon 连接串）
- 先 create_all 建全表，再按外键依赖顺序逐表搬运（全量覆盖式，目标须为空库）。
"""
from __future__ import annotations

import sys

from sqlalchemy import create_engine, text

from app.config import settings
from app.database import Base, engine as dst_engine
from app import models  # noqa: F401 确保全部表注册进 metadata

SRC_URL = "sqlite:///./aics.db"

src_engine = create_engine(SRC_URL)


def main() -> int:
    from sqlalchemy import text as _text

    with dst_engine.begin() as conn:
        conn.execute(_text('CREATE SCHEMA IF NOT EXISTS "aics"'))

    Base.metadata.create_all(bind=dst_engine)

    tables = Base.metadata.sorted_tables
    total_rows = 0
    with src_engine.connect() as src, dst_engine.begin() as dst:
        # 目标库必须为空，防止重复搬运撞主键
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
            # 分块插入，避免单条语句过大
            chunk = 500
            for i in range(0, len(rows), chunk):
                dst.execute(table.insert(), rows[i : i + chunk])
            total_rows += len(rows)
            print(f"  {table.name}: {len(rows)} 行")

    print(f"DONE: 共迁移 {total_rows} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
