"""表格形态判别的回归测试。

阈值是对着真实业务表标定的，改 THRESH 前先看 verify_table_layout.py 的输出。
"""
from app.tablelayout import (
    TRANSPOSED,
    ROW_WISE,
    classify,
    fieldness,
    split_grids,
    to_records,
)

# 转置表：行 = 属性，列 = 型号
BRIDGE = [
    ["", "WB730", "WB620E", "WB620F"],
    ["重量", "15.5KG", "13KG", "11.5KG"],
    ["传输距离", "5KM", "5KM", "15KM"],
    ["价格", "￥165", "￥145", "￥240"],
    ["配件", "24V PoE power,User Manual", "24V PoE power", "24V PoE power"],
]

# 行式表：列 = 字段，行 = 记录
ORDERS = [
    ["订单号", "日期", "金额", "状态"],
    ["A001", "2026-03-01", "¥120", "已发货"],
    ["A002", "2026-03-02", "¥80", "待发货"],
    ["A003", "2026-03-03", "¥240", "已发货"],
]


def test_transposed_detected():
    layout = classify(BRIDGE)
    assert layout.kind == TRANSPOSED
    # 判定为转置后，每个型号应拿到一条自足记录
    records = to_records(BRIDGE, layout)
    titles = [t for t, _ in records]
    assert "WB730" in titles
    wb730 = dict(records[titles.index("WB730")][1])
    assert wb730["重量"] == "15.5KG"
    assert wb730["价格"] == "￥165"


def test_row_wise_detected():
    layout = classify(ORDERS)
    assert layout.kind == ROW_WISE
    records = to_records(ORDERS, layout)
    assert len(records) == 3
    first = dict(records[0][1])
    assert first["订单号"] == "A001"
    assert first["金额"] == "¥120"


def test_fieldness_prefers_field_names_over_entity_codes():
    """同一把尺子量两维：字段名明显高于型号编号。"""
    assert fieldness(["重量", "价格", "传输距离"]) > fieldness(["WB730", "WB620E", "WB620F"])


def test_merged_cells_are_filled():
    """首列合并单元格（None）不会导致属性名丢失。"""
    grid = [
        ["参数", "X1", "X2"],
        ["重量", "1KG", "2KG"],
        [None, "5KM", "10KM"],
        ["价格", "￥10", "￥20"],
    ]
    layout = classify(grid)
    assert layout.kind == TRANSPOSED
    records = to_records(grid, layout)
    x1 = dict(records[0][1])
    assert x1["重量"] == "1KG"          # ffill 后第二行归入「重量」而不是被当成无标签行


def test_multi_block_sheet_is_split():
    """一个工作表里的两张表（中间隔空行）必须切成两个网格。"""
    rows = [
        ["型号", "库存"],
        ["X-1", "120"],
        [None, None],
        ["城市", "负责人"],
        ["广州", "张三"],
    ]
    assert len(split_grids(rows)) == 2
