"""表格形态判别验证脚本（离线可跑，不依赖外部服务）。

用法：python verify_table_layout.py [-v]
  -v 打印每个样本切出来的块文本

样本按真实业务表的形状构造：无线网桥（转置）、Ozon 模板（多行表头行式）、
订单表（行式）、合并单元格转置表、单表多区块、不规则稀疏表、单行表。
"""
from __future__ import annotations

import sys

from app.tablelayout import classify, parse_sheet, split_grids, to_records, render_blocks

# --------------------------------------------------------------------------- #
# 样本
# --------------------------------------------------------------------------- #
MODELS = ["WB730", "WB620E", "WB620F", "WB2500", "WB610H", "WB610F", "WB510C", "WB630"]

# 1) 转置参数表：行 = 属性，列 = 型号（无线网桥真实形状，值取自实际文件）
BRIDGE = [
    ["AI Smart Wireless Bridge/CPE"] + ["Outdoor 5G Gigabit Series"] * 4 + ["5G/450Mbps Series"] * 4,
    ["Model"] + MODELS,
    ["Chipset"] + ["MTK 7621+7612"] * 4 + ["MTK 7620+7612E"] * 4,
    ["Flash/RAM"] + ["8MB/64MB"] * 8,
    ["Frequency"] + ["5.8GHz"] * 8,
    ["PTP Distance"] + ["5KM", "5KM", "5KM", "15KM", "5KM", "5KM", "10KM", "5KM"],
    ["Wireless data rate"] + ["1000Mbps"] * 4 + ["900Mbps"] * 2 + ["450Mbps"] * 2,
    ["LAN ports"] + ["2*1000Mbps"] * 6 + ["1*1000Mbps+1*100Mbps"] * 2,
    ["POE ports"] + ["2*POE"] * 8,
    ["LED Display"] + ["With", "With", "With", "Without", "With", "With", "Without", "With"],
    ["Antenna"] + ["16dBi Panel Antenna"] * 3 + ["26dBi Panel Antenna"] + ["14dBi Panel Antenna"] * 4,
    ["Packing"] + ["8 Pairs in CTN", "10 Pairs in CTN"] * 4,
    ["CTN Size"] + ["430*315*540mm", "400*362*395mm", "380*330*390mm", "620*385*350mm",
                    "380*330*390mm", "465*310*425mm", "470*300*540mm", "405*300*470mm"],
    ["CTN weight"] + ["15.5KG", "13KG", "11.5KG", "11.2KG", "11.5KG", "11.5KG", "12KG", "14KG"],
    ["PRICE(RMB)"] + ["＜80PCS,￥165/PCS", "＜100PCS,￥145/PCS", "＜100PCS,￥145/PCS",
                      "＜100PCS,￥240/PCS", "＜100PCS,￥127/PCS", "＜100PCS,￥127/PCS",
                      "＜80PCS,￥153/PCS", "＜200PCS,￥70/PCS"],
    ["Funtional advantage"] + ["1、IP address management Settings 2、Point-to-multipoint extended camera monitoring range"] * 8,
    ["Accessory"] + ["24V PoE power,User Manual, Install accessory, LAN cable"] * 8,
]

# 2) 多行表头 + 行式表：第一行是分组名，第二行才是字段名（Ozon 模板真实形状）
OZON = [
    ["", "商品信息", "", "", "", "商品信息", "", "特性", "", "", ""],
    ["货号*", "商品名称", "非促销最高价格，CNY*", "划线价，CNY", "SKU", "毛重，克*",
     "包装宽度，毫米*", "类型*", "商品颜色", "品牌*", "简介"],
    ["FISH-001", "控鱼器 大号", "890", "1090", "SKU-A1", "300", "120", " fishing", "黑色", "AnglerPro",
     "控鱼器采用高强度铝合金材质，表面阳极氧化处理，适合淡海钓通用"],
    ["FISH-002", "控鱼器 小号", "690", "890", "SKU-A2", "203", "95", " fishing", "银色", "AnglerPro",
     "小巧便携，单手可操作，适合路亚钓法"],
    ["FISH-003", "钓鱼钳", "1200", "1500", "SKU-B1", "450", "150", " fishing", "蓝色", "AnglerPro",
     "多功能钓鱼钳，含剪线、取钩、压铅功能"],
]

# 3) 标准行式表
ORDERS = [
    ["订单号", "日期", "金额", "状态", "备注"],
    ["A001", "2026-03-01", "¥120", "已发货", "客户要求工作日送达"],
    ["A002", "2026-03-02", "¥80", "待发货", ""],
    ["A003", "2026-03-03", "¥240", "已发货", "需要发票"],
    ["A004", "2026-03-04", "¥65", "待发货", ""],
]

# 4) 转置表 + 首列合并单元格（后续行首列为 None）
MERGED = [
    ["参数"] + MODELS[:5],
    ["重量", "15.5KG", "13KG", "11.5KG", "11.2KG", "11.5KG"],
    [None, "5KM", "5KM", "5KM", "15KM", "5KM"],
    [None, "5.8GHz"] * 1 + ["5.8GHz"] * 4,
    ["价格", "￥165", "￥145", "￥145", "￥240", "￥127"],
]

# 5) 一个工作表里两张表，中间隔空行空列
MULTI = [
    ["型号", "库存", "单价", None, None, None],
    ["X-1", "120", "¥30", None, None, None],
    ["X-2", "80", "¥45", None, None, None],
    [None, None, None, None, None, None],
    ["城市", "仓位数", "负责人", None, None, None],
    ["广州", "12", "张三", None, None, None],
    ["深圳", "8", "李四", None, None, None],
]

# 6) 不规则稀疏表：缺值、长度不齐、夹了一行说明
MESSY = [
    ["产品", "规格", "适用范围", "备注"],
    ["网桥 A", "5KM", "园区监控", None],
    ["网桥 B", None, "电梯监控", "需另购支架"],
    ["说明：以上参数为实验室实测值，实际以现场环境为准", None, None, None],
    ["网桥 C", "10KM", "森林防火"],
]

# 7) 只有一行数据
SINGLE = [
    ["型号", "重量", "价格"],
    ["WB730", "15.5KG", "¥165"],
]

CASES = [
    ("无线网桥（转置参数表）", BRIDGE, "transposed"),
    ("Ozon 模板（多行表头+行式）", OZON, "row_wise"),
    ("订单表（行式）", ORDERS, "row_wise"),
    ("合并单元格转置表", MERGED, "transposed"),
    ("单表多区块", MULTI, "row_wise"),
    ("不规则稀疏表", MESSY, "row_wise"),
    ("单行数据表", SINGLE, None),
]


def main() -> int:
    verbose = "-v" in sys.argv
    print("=" * 78)
    print("表格形态判别验证")
    print("=" * 78)

    failed = 0
    for name, rows, expect in CASES:
        blocks, warnings = parse_sheet(rows, sheet_name=name[:6])
        grids = split_grids(rows)
        # 主网格判定结果（多区块时取第一个）
        layout = classify(grids[0]) if grids else None
        kind = layout.kind if layout else "flat"
        ok = "OK " if (expect is None or kind == expect) else "FAIL"
        if ok == "FAIL":
            failed += 1
        sig = layout.signals if layout else {}
        print(f"\n[{ok}] {name}")
        print(f"   判定={kind:<11} 期望={expect or '任意':<11} score={layout.score if layout else 0}")
        if layout:
            print(f"   row_h={layout.row_h}  col_h={layout.col_h}  delta={sig.get('delta')}"
                  f"  f_header={sig.get('f_header')}  f_col0={sig.get('f_col0')}"
                  f"  header_dup={sig.get('header_dup')}")
            print(f"   表头={layout.names[:4]}{' ...' if len(layout.names) > 4 else ''}")
        print(f"   区块数={len(grids)}  文本块数={len(blocks)}  最长块={max((len(b) for b in blocks), default=0)}字")
        for w in warnings:
            print(f"   ! {w}")
        if verbose:
            for i, b in enumerate(blocks[:4]):
                print(f"   --- block {i} ({len(b)}字) ---")
                print("   " + b[:220].replace("\n", "\n   "))
            if len(blocks) > 4:
                print(f"   ... 其余 {len(blocks) - 4} 块省略")

    print("\n" + "=" * 78)
    print(f"结论：{len(CASES) - failed}/{len(CASES)} 通过")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
