"""数据库连接与会话。默认 SQLite 开箱即用，生产换 PostgreSQL 即可。"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


connect_args: dict = {}
engine_kwargs: dict = {"pool_pre_ping": True, "future": True}

if settings.is_sqlite:
    connect_args["check_same_thread"] = False
else:
    engine_kwargs.update({"pool_size": 10, "max_overflow": 20, "pool_recycle": 1800})
    # Neon 等托管库走 -pooler 端点 = PgBouncer 事务池，不支持会话级预编译语句；
    # psycopg3 默认执行 5 次后自动 PREPARE，会撞 "prepared statement does not exist"。
    # 注意只能在这里以整型传入——写在 URL query 里会变成字符串导致驱动内部报错。
    if settings.database_url.startswith("postgresql"):
        connect_args["prepare_threshold"] = 0

engine = create_engine(settings.database_url, connect_args=connect_args, **engine_kwargs)

if settings.is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - 驱动回调
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=8000")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每请求一个会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """脚本 / 后台任务用。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  确保模型注册

    _ensure_schema()
    Base.metadata.create_all(bind=engine)
    _ensure_new_columns()


def _ensure_schema() -> None:
    """PG 且连接串带 options=-csearch_path=<schema> 时，先确保该 schema 存在。

    Neon 等共享库的 public schema 可能有历史残留表与我们撞名，独立 schema 隔离；
    不显式建的话新环境首次启动会直接报 schema 不存在。
    """
    if settings.is_sqlite:
        return
    import re

    from sqlalchemy import text

    match = re.search(r"search_path(?:%3D|=)([A-Za-z_]\w*)", settings.database_url)
    if not match:
        return
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{match.group(1)}"'))


# create_all 只建新表、不会给已有表加新列。这里维护「后加的列」清单，
# 启动时对已存在的表逐列 ALTER ADD（仅 SQLite；换 PG 时走正式迁移）。
_NEW_COLUMNS: dict[str, list[tuple[str, str, str]]] = {
    "ai_employees": [
        ("humanize_enabled", "BOOLEAN", "0"),
        ("reply_delay_seconds", "INTEGER", "0"),
        ("reply_delay_min", "INTEGER", "0"),
        ("reply_delay_max", "INTEGER", "0"),
        ("split_reply_enabled", "BOOLEAN", "0"),
        ("split_reply_max", "INTEGER", "3"),
        ("split_reply_interval_ms", "INTEGER", "800"),
        ("split_interval_min_ms", "INTEGER", "800"),
        ("split_interval_max_ms", "INTEGER", "800"),
        ("stop_reply_enabled", "BOOLEAN", "0"),
        ("stop_reply_rounds", "INTEGER", "0"),
        ("stop_reply_message", "VARCHAR(500)", "''"),
        ("stop_condition_groups", "TEXT", "'[]'"),
        ("stop_time_enabled", "BOOLEAN", "0"),
        ("stop_time_rules", "TEXT", "'[]'"),
    ],
}


def _ensure_new_columns() -> None:
    if not settings.is_sqlite:
        return
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _NEW_COLUMNS.items():
            if table not in insp.get_table_names():
                continue  # 全新部署由 create_all 直接建全
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl, default in cols:
                if name in existing:
                    continue
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl} NOT NULL DEFAULT {default}")
                )
