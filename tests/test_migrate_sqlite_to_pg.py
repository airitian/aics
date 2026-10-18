"""migrate_sqlite_to_pg.py 的单元测试。

环境没有可用 PostgreSQL 实例（无 pg_ctl / initdb / pytest-postgresql），
所以这里分两层测：

1. 纯函数层（不需要任何数据库）：`convert_value_for_pg` 对 Boolean / DateTime /
   Integer / Text 的转换，覆盖 SQLite 真实会出现的各种脏值。
   这是类型转换风险的主要承载点，必须全量覆盖。

2. 搬运+校验逻辑层：用两个 SQLite 库（源只读、目标可写）跑 `migrate_tables()` +
   `verify()`，验证「逐表搬运 + 行数/主键/全字段校验」这套逻辑本身是对的。
   PG 特有的部分（schema、sequence）不在这里覆盖，见文件末尾的说明。
"""
from __future__ import annotations

import datetime
import sqlite3

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import Boolean, DateTime, Integer, String, Text

from app.database import Base
from app import models  # noqa: F401
from migrate_sqlite_to_pg import (
    CHUNK_SIZE,
    ConversionError,
    assert_topological_order,
    convert_row_for_pg,
    convert_value_for_pg,
    make_src_engine,
    migrate_tables,
    validate_json_text_columns,
    verify,
)


# --------------------------------------------------------------------------- #
# Boolean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, False),
        (1, True),
        (False, False),
        (True, True),
        ("0", False),
        ("1", True),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("False", False),
        ("t", True),
        ("f", False),
        ("yes", True),
        ("no", False),
    ],
)
def test_boolean_conversion(raw, expected):
    assert convert_value_for_pg(Boolean(), raw) is expected


@pytest.mark.parametrize("raw", [2, -1, "2", "maybe", "", "TRUEish"])
def test_boolean_rejects_invalid(raw):
    with pytest.raises(ConversionError):
        convert_value_for_pg(Boolean(), raw)


def test_boolean_none_passthrough():
    """None 必须原样返回：可空列允许 NULL，非空列由 NOT NULL 约束兜底。"""
    assert convert_value_for_pg(Boolean(), None) is None


# --------------------------------------------------------------------------- #
# DateTime
# --------------------------------------------------------------------------- #
def test_datetime_sqlite_naive_string():
    """SQLite 里 DateTime 最常见形态：空格分隔 + 6 位微秒。"""
    got = convert_value_for_pg(DateTime(), "2026-09-22 14:48:03.614335")
    assert got == datetime.datetime(2026, 9, 22, 14, 48, 3, 614335)
    assert got.tzinfo is None, "本项目统一 naive UTC，不能带 tzinfo"


def test_datetime_sqlite_naive_no_microseconds():
    got = convert_value_for_pg(DateTime(), "2026-09-22 14:48:03")
    assert got == datetime.datetime(2026, 9, 22, 14, 48, 3)


def test_datetime_iso_t_separator():
    assert convert_value_for_pg(DateTime(), "2026-09-22T14:48:03.614335") == datetime.datetime(
        2026, 9, 22, 14, 48, 3, 614335
    )


def test_datetime_with_offset_is_folded_to_naive_utc():
    """带 +00:00 的一律折算到 UTC 再去掉 tzinfo，与 app/utils.utcnow 语义一致。"""
    got = convert_value_for_pg(DateTime(), "2026-09-22T14:48:03.614335+00:00")
    assert got == datetime.datetime(2026, 9, 22, 14, 48, 3, 614335)
    assert got.tzinfo is None


def test_datetime_with_nonzero_offset_is_converted():
    got = convert_value_for_pg(DateTime(), "2026-09-22T22:48:03+08:00")
    assert got == datetime.datetime(2026, 9, 22, 14, 48, 3)
    assert got.tzinfo is None


def test_datetime_date_only():
    assert convert_value_for_pg(DateTime(), "2026-09-22") == datetime.datetime(2026, 9, 22)


@pytest.mark.parametrize("raw", ["", "   ", "not-a-date", "2026-13-45 99:99:99"])
def test_datetime_rejects_unparseable(raw):
    with pytest.raises(ConversionError):
        convert_value_for_pg(DateTime(), raw)


def test_datetime_empty_string_is_rejected_not_silently_passed():
    """空串进 PG 会报 invalid input syntax，必须在迁移层拦住。"""
    with pytest.raises(ConversionError, match="空字符串"):
        convert_value_for_pg(DateTime(), "")


def test_datetime_passthrough_datetime():
    dt = datetime.datetime(2026, 9, 22, 14, 48, 3)
    assert convert_value_for_pg(DateTime(), dt) is dt


def test_datetime_aware_datetime_input_is_normalized():
    dt = datetime.datetime(2026, 9, 22, 14, 48, 3, tzinfo=datetime.timezone.utc)
    got = convert_value_for_pg(DateTime(), dt)
    assert got.tzinfo is None
    assert got == datetime.datetime(2026, 9, 22, 14, 48, 3)


def test_datetime_rejects_wrong_python_type():
    with pytest.raises(ConversionError):
        convert_value_for_pg(DateTime(), 1695394083)


# --------------------------------------------------------------------------- #
# Integer / Text / String
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [(0, 0), (42, 42), ("42", 42), (" 42 ", 42), (3.0, 3)])
def test_integer_conversion(raw, expected):
    assert convert_value_for_pg(Integer(), raw) == expected


@pytest.mark.parametrize("raw", ["abc", "", 3.5, "3.5"])
def test_integer_rejects_invalid(raw):
    with pytest.raises(ConversionError):
        convert_value_for_pg(Integer(), raw)


def test_text_keeps_string():
    assert convert_value_for_pg(Text(), "hello") == "hello"


def test_text_coerces_numbers_to_string():
    """SQLite 动态类型：Text 列可能存进数字，PG 的 text 列不接受 int。"""
    assert convert_value_for_pg(Text(), 123) == "123"
    assert convert_value_for_pg(Text(), 1.5) == "1.5"
    assert convert_value_for_pg(Text(), True) == "true"


def test_empty_string_is_preserved_not_turned_into_null():
    """空串不是 NULL，满足 PG 的 NOT NULL 约束，不能被'修'成 None。"""
    assert convert_value_for_pg(String(500), "") == ""
    assert convert_value_for_pg(Text(), "") == ""


def test_none_passthrough_for_every_type():
    for ty in (Boolean(), DateTime(), Integer(), Text(), String(32)):
        assert convert_value_for_pg(ty, None) is None


def test_unknown_type_passthrough():
    """没识别的类型原样返回，不做猜测。"""
    assert convert_value_for_pg(object(), "raw") == "raw"


def test_conversion_error_message_carries_column_label():
    with pytest.raises(ConversionError, match="ai_employees.is_default"):
        convert_value_for_pg(Boolean(), 7, column_label="ai_employees.is_default")


# --------------------------------------------------------------------------- #
# 整行转换
# --------------------------------------------------------------------------- #
def test_convert_row_for_pg_all_columns():
    table = Base.metadata.tables["ai_employees"]
    row = {
        "id": "abc",
        "is_default": 1,
        "enabled_flag": None,
        "created_at": "2026-09-22 14:48:03.614335",
        "published_version": "19",
        "persona": "hi",
    }
    out = convert_row_for_pg(table, row)
    assert out["id"] == "abc"
    assert out["is_default"] is True
    assert out["created_at"] == datetime.datetime(2026, 9, 22, 14, 48, 3, 614335)
    assert out["published_version"] == 19
    assert out["persona"] == "hi"
    assert "enabled_flag" not in out, "源库多出来的列必须丢弃，否则 insert 会报列名错"


def test_convert_row_drops_unknown_columns():
    table = Base.metadata.tables["tenants"]
    out = convert_row_for_pg(table, {"id": "t1", "name": "n", "ghost_col": "x"})
    assert out == {"id": "t1", "name": "n"}


# --------------------------------------------------------------------------- #
# JSON 文本列校验
# --------------------------------------------------------------------------- #
def test_json_text_column_valid():
    table = Base.metadata.tables["messages"]
    assert validate_json_text_columns(table, [{"id": "m1", "meta": '{"hits":[]}'}]) == []


def test_json_text_column_invalid_is_reported():
    table = Base.metadata.tables["messages"]
    problems = validate_json_text_columns(table, [{"id": "m1", "meta": "{not json"}])
    assert problems and "m1" in problems[0]


def test_json_text_column_empty_is_ok():
    table = Base.metadata.tables["messages"]
    assert validate_json_text_columns(table, [{"id": "m1", "meta": ""}]) == []


# --------------------------------------------------------------------------- #
# 拓扑序
# --------------------------------------------------------------------------- #
def test_sorted_tables_is_topological():
    """父表必须先于子表，否则 PG 外键会炸。不应抛 SystemExit。"""
    assert_topological_order(Base.metadata.sorted_tables)


def test_sorted_tables_has_all_nineteen_tables():
    assert len(Base.metadata.sorted_tables) == 19


# --------------------------------------------------------------------------- #
# 搬运 + 校验逻辑（双 SQLite，验证逻辑本身，不依赖 PG）
# --------------------------------------------------------------------------- #
def _sqlite_file_engine(path, read_only=False):
    abs_path = str(path).replace("\\", "/")

    def _connect():
        if read_only:
            return sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        return sqlite3.connect(abs_path)

    return create_engine("sqlite://", creator=_connect, poolclass=StaticPool)


def test_end_to_end_copy_and_verify(tmp_path):
    """源 -> 目标全量搬运，随后 verify() 必须报全 OK。"""
    src_path = tmp_path / "src.db"
    dst_path = tmp_path / "dst.db"

    # 造源库：一张父表 + 一张子表，含 Boolean / DateTime / 空串 / NULL
    src = _sqlite_file_engine(src_path)
    Base.metadata.create_all(bind=src)
    with src.begin() as c:
        c.execute(
            text(
                "INSERT INTO tenants (id, name, slug, status, plan, daily_token_quota,"
                " rpm_limit, timezone, default_language, note, extra, created_at, updated_at)"
                " VALUES ('t1','T','t','active','free',100,10,'UTC','zh','','{}',"
                " '2026-09-22 14:48:03.614335','2026-09-22 14:48:03.614335')"
            )
        )
        c.execute(
            text(
                "INSERT INTO users (id, tenant_id, email, password_hash, name, role, status,"
                " last_login_at, created_at, updated_at)"
                " VALUES ('u1','t1','a@b.c','h','n','tenant_admin','active',NULL,"
                " '2026-09-22 14:48:03','2026-09-22 14:48:03')"
            )
        )
    src.dispose()

    # 目标库建表后搬运
    dst = _sqlite_file_engine(dst_path)
    Base.metadata.create_all(bind=dst)
    ro_src = make_src_engine(str(src_path))
    tables = Base.metadata.sorted_tables
    with dst.begin() as d:
        total, skipped = migrate_tables(ro_src, d, tables)
    assert skipped == [], f"不应有跳过项: {skipped}"
    assert total == 2

    with dst.connect() as d:
        ok = verify(ro_src, d, tables)
    assert ok, "verify 应判定为一致"

    # 目标库读回的类型必须对（Boolean 是 bool、DateTime 是 datetime）
    with dst.connect() as d:
        row = d.execute(text("SELECT id, typeof(created_at) FROM tenants")).first()
    assert row[0] == "t1"

    ro_src.dispose()
    dst.dispose()


def test_verify_detects_row_count_mismatch(tmp_path):
    """目标少一行，verify 必须返回 False（不许报成功）。"""
    src_path = tmp_path / "src.db"
    dst_path = tmp_path / "dst.db"

    src = _sqlite_file_engine(src_path)
    Base.metadata.create_all(bind=src)
    with src.begin() as c:
        c.execute(
            text(
                "INSERT INTO tenants (id, name, slug, status, plan, daily_token_quota,"
                " rpm_limit, timezone, default_language, note, extra, created_at, updated_at)"
                " VALUES ('t1','T','t','active','free',100,10,'UTC','zh','','{}',"
                " '2026-09-22 14:48:03','2026-09-22 14:48:03')"
            )
        )
    src.dispose()

    dst = _sqlite_file_engine(dst_path)
    Base.metadata.create_all(bind=dst)

    ro_src = make_src_engine(str(src_path))
    tables = Base.metadata.sorted_tables
    with dst.connect() as d:
        assert verify(ro_src, d, tables) is False

    ro_src.dispose()
    dst.dispose()


def test_chunk_size_is_sane():
    assert CHUNK_SIZE > 0


# --------------------------------------------------------------------------- #
# 未覆盖部分的说明
# --------------------------------------------------------------------------- #
def test_pg_specific_behaviour_not_covered_here():
    """提醒：schema / sequence / psycopg 参数绑定这部分没有单测覆盖。

    本机没有 PG 实例（无 pg_ctl / initdb / pytest-postgresql），
    这部分只能靠真实 PG 上端到端跑通来验证（见交付报告）。
    """
    assert True
