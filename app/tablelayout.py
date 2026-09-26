"""表格形态识别与结构化切分。

为什么需要这个模块
------------------
同一份 xlsx 可能是「一行一条记录」的名单表，也可能是「一列一个型号」的参数对照表，
两种表的正确切法完全相反。按扩展名或按行列数猜都会翻车（实测无线网桥是 24 行 × 18 列，
行数反而大于列数），所以只能看**哪一维是字段**：

- 行式表：列 = 字段 → 列内同质（日期列全是日期）、行内混杂
- 转置表：行 = 字段 → 行内同质（重量行全是重量）、列内混杂

不规则表不硬猜：走「保守按行 + 表名表头入块 + warning」的兜底链，
宁可上下文不全，也不要把型号和重量错位到两个块里。

所有阈值集中在 THRESH，改一处即可调。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ROW_WISE = "row_wise"
TRANSPOSED = "transposed"
UNKNOWN = "unknown"
FLAT = "flat"

THRESH = {
    "blank_rate_for_ffill": 0.30,   # 首列空白率超过此值 → 认为有合并单元格，启用前向填充
    "min_cells_per_row": 2,         # 非空单元格少于此值的行直接丢弃
    "min_rows": 2,
    "min_cols": 2,
    "delta_full": 0.40,             # 同质度差值达到此值时打满（次要信号）
    "w_fieldness": 1.50,            # 「哪一维更像字段名」——主信号
    "w_homogeneity": 0.40,          # 行/列同质度差——次要信号
    "score_transposed": 0.50,
    "score_row_wise": -0.50,
    "header_fill_ratio": 0.80,      # 判定「表头行」所需的填充率（相对最大填充率）
    "header_scan": 4,               # 表头最多往前看几行
    "key_field_count": 2,           # 超长记录拆分时复制到每块的主键字段数
}

_BLANK_TOKENS = {"", "-", "--", "/", "\\", "N/A", "NA", "—", "－", "无"}
_CURRENCY = "¥￥$€£"
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?%?$")
_DATE_RE = re.compile(r"^(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$")

# 字段名高频词。单个词必漏（跨语言、跨行业），但本模块是**拿同一把尺子量两维**再做差，
# 所以词表漏词不会系统性偏向某一侧——这正是它比绝对阈值稳健的原因。
_FIELD_WORDS = (
    "model", "型号", "chipset", "芯片", "flash", "ram", "内存", "存储",
    "weight", "重量", "毛重", "净重", "size", "尺寸", "长度", "宽度", "高度",
    "price", "价格", "单价", "金额", "总价", "成本",
    "ports", "端口", "frequency", "频率", "power", "功率", "poe", "电压",
    "rate", "速率", "distance", "距离", "antenna", "天线",
    "packing", "包装", "accessory", "配件", "ctn", "net", "gross",
    "display", "led", "switch", "材质", "颜色", "品牌",
    "日期", "时间", "状态", "备注", "说明", "名称", "编号", "货号", "订单",
    "类型", "简介", "数量", "库存", "单位", "规格", "适用范围", "参数",
    "地址", "电话", "客户", "城市", "区域", "省份", "负责人",
    "sku", "ean", "barcode", "stock", "qty", "unit", "color", "brand",
    "type", "name", "date", "status", "remark", "note", "desc", "price",
)

# 实体名（型号/编号）常见形态：字母前缀 + 数字尾巴，如 WB730 / A001 / FISH-001 / X-1。
# 命中它说明这一维「像实体」而不是「像字段名」。
_MODEL_RE = re.compile(r"^[A-Za-z]{1,8}[-_ ]?\d{1,}[A-Za-z0-9]*$")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def is_blank(v) -> bool:
    if v is None:
        return True
    return str(v).strip() in _BLANK_TOKENS


def _rect(rows: list[list]) -> list[list]:
    width = max((len(r) for r in rows), default=0)
    return [list(r) + [None] * (width - len(r)) for r in rows]


def _nonblank_count(row) -> int:
    return sum(0 if is_blank(v) else 1 for v in row)


def _avg_len(row) -> float:
    vals = [str(v).strip() for v in row if not is_blank(v)]
    return sum(len(v) for v in vals) / len(vals) if vals else 0.0


def _drop_blank_cols(grid: list[list]) -> list[list]:
    if not grid:
        return grid
    keep = [j for j in range(len(grid[0])) if any(not is_blank(r[j]) for r in grid)]
    return [[r[j] for j in keep] for r in grid]


# --------------------------------------------------------------------------- #
# 不规则表预处理
# --------------------------------------------------------------------------- #
def split_grids(rows: list[list]) -> list[list[list]]:
    """按全空行 / 全空列切分成多个独立网格。

    一个工作表里塞了两张表（中间隔空行空列）是很常见的形态，
    不切开会得到一张稀疏大表，同质度被空白严重稀释。
    """
    rows = _rect(rows)
    if not rows:
        return []

    bands: list[list[list]] = []
    cur: list[list] = []
    for row in rows:
        if _nonblank_count(row) == 0:
            if cur:
                bands.append(cur)
                cur = []
        else:
            cur.append(row)
    if cur:
        bands.append(cur)

    grids: list[list[list]] = []
    for band in bands:
        cols = [j for j in range(len(band[0])) if any(not is_blank(r[j]) for r in band)]
        if not cols:
            continue
        groups: list[list[int]] = []
        g = [cols[0]]
        for j in cols[1:]:
            if j == g[-1] + 1:
                g.append(j)
            else:
                groups.append(g)
                g = [j]
        groups.append(g)
        for group in groups:
            sub = [[r[j] for j in group] for r in band]
            sub = [r for r in sub if _nonblank_count(r) >= THRESH["min_cells_per_row"]]
            sub = _drop_blank_cols(sub)
            if len(sub) >= THRESH["min_rows"] and len(sub[0]) >= THRESH["min_cols"]:
                grids.append(sub)
    return grids


def ffill_first_col(grid: list[list]) -> list[list]:
    """合并单元格修复：首列向下前向填充。

    Excel 里合并单元格只有首格有值，openpyxl 读到的是 None；
    不填充的话转置表的属性名会大面积缺失，行同质度被拉低。
    """
    out = [list(r) for r in grid]
    last = None
    for row in out:
        if is_blank(row[0]):
            if last is not None:
                row[0] = last
        else:
            last = row[0]
    return out


def _blank_rate_col(grid: list[list], j: int = 0) -> float:
    if not grid:
        return 0.0
    return sum(1 for r in grid if is_blank(r[j])) / len(grid)


# --------------------------------------------------------------------------- #
# 同质度
# --------------------------------------------------------------------------- #
def cell_type(v) -> str | None:
    if is_blank(v):
        return None
    s = str(v).strip()
    if _DATE_RE.match(s):
        return "date"
    t = s.lstrip(_CURRENCY).replace(",", "")
    if _NUM_RE.match(t):
        return "num"
    if s[0].isdigit():
        return "measure"          # 15.5KG / 5KM / 8MB/64MB / 2*1000Mbps / 430*315*540mm
    if len(s) <= 24 and s.count(" ") <= 3 and "，" not in s and "。" not in s:
        return "code"             # WB730 / MTK 7621+7612 / With / POE 24V
    return "text"


def purity(values) -> float:
    """类型纯度 = 占比最高的类型 / 非空单元格数。"""
    types = [t for t in (cell_type(v) for v in values) if t]
    if not types:
        return 0.0
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / len(types)


def homogeneity(data: list[list]) -> tuple[float, float]:
    """返回 (行同质度, 列同质度)。两边都排除首列——首列在两种表里都可能是标签列。"""
    body = [r[1:] for r in data]
    if not body or not body[0]:
        return 0.0, 0.0
    row_h = sum(purity(r) for r in body) / len(body)
    cols = [[r[j] for r in body] for j in range(len(body[0]))]
    col_h = sum(purity(c) for c in cols) / len(cols)
    return row_h, col_h


def fieldness(values) -> float:
    """这一组取值「像字段名」的程度，0~1。

    四个特征：含量纲/字段词、短、唯一、不像型号编号（最后一条是负分）。
    """
    vals = [str(v).strip() for v in values if not is_blank(v)]
    if not vals:
        return 0.0
    n = len(vals)
    kw = sum(1 for v in vals if any(w in v.lower() for w in _FIELD_WORDS)) / n
    short = sum(1 for v in vals if len(v) <= 10) / n
    uniq = len(set(vals)) / n
    model = sum(1 for v in vals if _MODEL_RE.match(v)) / n
    raw = 1.2 * kw + 0.5 * short + 0.3 * uniq - 0.8 * model
    return max(0.0, min(1.0, raw))


def name_likeness(values) -> float:
    """这一组取值「像实体名（型号/编号）」的程度——用于转置表定位实体名所在行。"""
    vals = [str(v).strip() for v in values if not is_blank(v)]
    if not vals:
        return 0.0
    n = len(vals)
    model = sum(1 for v in vals if _MODEL_RE.match(v)) / n
    uniq = len(set(vals)) / n
    short = 1.0 if sum(len(v) for v in vals) / n <= 12 else 0.0
    return model + 0.3 * uniq + 0.3 * short


# --------------------------------------------------------------------------- #
# 表头定位（支持分组行 + 字段行的多行表头）
# --------------------------------------------------------------------------- #
def _header_span(grid: list[list]) -> tuple[int, int]:
    """返回表头行区间 [start, end]（含）。"""
    fills = [_nonblank_count(r) for r in grid]
    if not fills:
        return 0, 0
    max_fill = max(fills) or 1
    limit = min(THRESH["header_scan"], len(fills))
    start = 0
    for i in range(limit):
        if fills[i] >= THRESH["header_fill_ratio"] * max_fill:
            start = i
            break
    # 低填充的前一行通常是分组行（如「商品信息」「特性」），作为层级前缀保留
    lo = start
    if start > 0 and fills[start - 1] > 0:
        lo = start - 1
    else:
        for i in range(start + 1, min(start + 2, len(fills))):
            if fills[i] >= THRESH["header_fill_ratio"] * max_fill and _avg_len(grid[i]) <= _avg_len(grid[start]) * 1.2:
                lo = start
                break
    return lo, start


def _entity_row(grid: list[list]) -> tuple[int, int]:
    """转置表定位实体名所在行：在前 3 行里挑「最像实体名」的一行。

    无线网桥这类表，第 0 行是系列分组（Outdoor 5G Gigabit Series）、第 1 行才是型号名，
    两行都要保留，拼成『系列 > 型号』。
    """
    limit = min(3, len(grid))
    best, best_score = 0, -1.0
    for i in range(limit):
        s = name_likeness([v for v in grid[i][1:]])
        if s > best_score:
            best, best_score = i, s
    lo = best
    if best > 0:
        above = [v for v in grid[best - 1][1:] if not is_blank(v)]
        if above and len(set(str(v).strip() for v in above)) / len(above) < 0.6:
            lo = best - 1          # 上方是低唯一度的分组行，作为层级前缀保留
    return lo, best


def header_names(grid: list[list], span: tuple[int, int]) -> list[str]:
    """把多行表头拼成层级列名：『商品信息 > 毛重，克*』。"""
    lo, hi = span
    names: list[str] = []
    width = len(grid[0])
    for j in range(width):
        parts = [str(grid[i][j]).strip() for i in range(lo, hi + 1) if not is_blank(grid[i][j])]
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        names.append(" > ".join(seen) if seen else f"列{j + 1}")
    return names


# --------------------------------------------------------------------------- #
# 形态判定
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    kind: str
    score: float = 0.0
    row_h: float = 0.0
    col_h: float = 0.0
    header_span: tuple[int, int] = (0, 0)
    names: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def classify(grid: list[list]) -> Layout:
    grid = _drop_blank_cols(_rect(grid))
    if len(grid) < THRESH["min_rows"] or len(grid[0]) < THRESH["min_cols"]:
        return Layout(FLAT, warnings=["不构成表格（行或列不足），按普通文本处理"])

    ffilled = False
    if _blank_rate_col(grid, 0) >= THRESH["blank_rate_for_ffill"]:
        grid = ffill_first_col(grid)
        ffilled = True

    # 先按填充率粗定表头行，再取数据区。顺序不能反：
    # 多行表头（分组行 + 字段行）若不先定位，字段行会被当成数据行，f_col0 被稀释。
    fill_span = _header_span(grid)
    data = grid[fill_span[1] + 1:]
    row_h, col_h = homogeneity(data)
    delta = row_h - col_h

    # 主信号：同一把尺子量「表头行」与「首列」，谁更像字段名，谁就是字段轴。
    # 首列更像字段名 → 列是实体 → 转置表；表头行更像 → 行是实体 → 行式表。
    f_header = fieldness([v for v in grid[fill_span[1]][1:]])
    f_col0 = fieldness([r[0] for r in data])
    score = THRESH["w_fieldness"] * (f_col0 - f_header)
    # 次要信号：同质度差。产品高度相似时（多行参数相同）这个信号会被稀释，故只作辅助。
    score += THRESH["w_homogeneity"] * max(-1.0, min(1.0, delta / THRESH["delta_full"]))

    if score >= THRESH["score_transposed"]:
        kind = TRANSPOSED
    elif score <= THRESH["score_row_wise"]:
        kind = ROW_WISE
    else:
        kind = UNKNOWN

    span = _entity_row(grid) if kind == TRANSPOSED else fill_span
    header_vals = [v for v in grid[span[1]][1:] if not is_blank(v)]
    uniq = len({str(v).strip() for v in header_vals})
    dup = 1 - (uniq / len(header_vals)) if header_vals else 0.0
    warnings: list[str] = []
    if ffilled:
        warnings.append("检测到首列存在合并单元格，已向下填充")
    if dup >= 0.30:
        warnings.append("首行疑似分组表头（非字段名），已按多行表头合并")
    if kind == UNKNOWN:
        warnings.append("表格形态无法判定，已保守按行切分；可在文件详情切换为按列解析")

    return Layout(
        kind=kind,
        score=round(score, 3),
        row_h=round(row_h, 3),
        col_h=round(col_h, 3),
        header_span=span,
        names=header_names(grid, span),
        signals={
            "delta": round(delta, 3),
            "f_header": round(f_header, 3),
            "f_col0": round(f_col0, 3),
            "header_dup": round(dup, 3),
        },
        warnings=warnings,
    )


def _is_blank(v) -> bool:
    return is_blank(v)


# --------------------------------------------------------------------------- #
# 输出记录块
# --------------------------------------------------------------------------- #
def to_records(grid: list[list], layout: Layout) -> list[tuple[str, dict]]:
    """返回 [(记录标题, {字段: 值})]。"""
    grid = _drop_blank_cols(_rect(grid))
    if _blank_rate_col(grid, 0) >= THRESH["blank_rate_for_ffill"]:
        grid = ffill_first_col(grid)

    names = layout.names or header_names(grid, layout.header_span)
    _, hi = layout.header_span
    data = grid[hi + 1:]

    out: list[tuple[str, dict]] = []
    if layout.kind == TRANSPOSED:
        # 列 = 实体：每个实体一条记录，字段来自每行首列
        for j in range(1, len(grid[0])):
            title = names[j].strip() if j < len(names) and names[j].strip() else f"第{j}列"
            rec: dict[str, object] = {}
            for row in data:
                key = str(row[0]).strip() if not is_blank(row[0]) else ""
                if not key:
                    continue
                val = str(row[j]).strip()
                if is_blank(val):
                    continue
                if key in rec and rec[key] != val:
                    # 首列 ffill 后可能出现同名属性（源表的重复属性块），
                    # 静默覆盖会丢信息（实测出现过「重量=5.8GHz」），改为拼接并告警
                    rec[key] = f"{rec[key]}；{val}"
                else:
                    rec[key] = val
            if rec:
                out.append((title, rec))
    else:
        # 行 = 记录（unknown 也走这条，保守但不会错位）
        for idx, row in enumerate(data, start=1):
            rec = {}
            for j, name in enumerate(names):
                if j >= len(row) or is_blank(row[j]):
                    continue
                rec[name] = str(row[j]).strip()
            if not rec:
                continue
            # 标题取行内第一个非空值（通常是货号/编号/名称），比「第N条」自足得多
            first = next((str(v).strip() for v in row if not is_blank(v)), f"第{idx}条")
            out.append((first, rec))
    return out


def render_blocks(
    records: list[tuple[str, dict]],
    *,
    sheet_name: str = "",
    max_chars: int = 600,
    key_fields: list[str] | None = None,
) -> list[str]:
    """把记录渲染成自足文本块；单条超长时按字段分组拆块并复制主键字段。"""
    head = f"【{sheet_name}】" if sheet_name else ""
    keys = key_fields or _guess_key_fields(records)
    blocks: list[str] = []

    for title, rec in records:
        items = [(k, v) for k, v in rec.items() if not is_blank(v)]
        if not items:
            continue
        prefix = f"{head}{title}："

        def render(parts: list[tuple[str, str]]) -> str:
            return prefix + "；".join(f"{k}: {v}" for k, v in parts)

        whole = render(items)
        if len(whole) <= max_chars:
            blocks.append(whole)
            continue

        # 超长记录：主键字段复制到每一块，保证每块都自足
        key_part = [kv for kv in items if kv[0] in keys]
        rest = [kv for kv in items if kv[0] not in keys]
        base = len("；".join(f"{k}: {v}" for k, v in key_part)) + 2
        groups: list[list[tuple[str, str]]] = []
        buf: list[tuple[str, str]] = []
        used = 0
        for kv in rest:
            add = len(f"{kv[0]}: {kv[1]}") + 1
            if buf and base + used + add > max_chars:
                groups.append(buf)
                buf, used = [], 0
            buf.append(kv)
            used += add
        if buf:
            groups.append(buf)
        for group in groups:
            blocks.append(render(key_part + group))
    return blocks


def _guess_key_fields(records: list[tuple[str, dict]]) -> list[str]:
    """主键字段 = 最靠前的若干列（表格前几列通常是编号/名称）。"""
    if not records:
        return []
    names = list(records[0][1].keys())
    return names[: THRESH["key_field_count"]]


def _leading_title_cells(rows: list[list]) -> list[str]:
    """提取表格数据区之前的「独值行」文本（通常是合并单元格的大标题）。

    为什么需要：split_grids 的 min_cells_per_row 会把只有一个非空格的行当噪声丢掉，
    但整表第一行往往是品牌/系列标题（实测无线网桥表第 1 行是
    「AI Smart Wireless Bridge/CPE」）——丢了它，品牌类提问（"AI Smart 是什么"）
    在知识库里就没有任何可 grounding 的内容，模型只能靠猜。
    只认数据区之前连续的独值行，且最多 3 行：单列长列表（每行都只有 1 格）
    不会被误判成标题。
    """
    cells: list[str] = []
    for row in rows:
        if _nonblank_count(row) >= THRESH["min_cells_per_row"]:
            break
        vals = [str(v).strip() for v in row if not is_blank(v)]
        if vals:
            cells.append(vals[0])
    return cells if 0 < len(cells) <= 3 else []


def parse_sheet(rows: list[list], *, sheet_name: str = "", max_chars: int = 600) -> tuple[list[str], list[str]]:
    """工作表入口：切分 → 判定 → 渲染。返回 (文本块列表, warning 列表)。"""
    blocks: list[str] = []
    warnings: list[str] = []
    rows = _rect([list(r) for r in rows])
    title_cells = _leading_title_cells(rows)
    if title_cells:
        head = f"【{sheet_name}·表标题】" if sheet_name else "【表标题】"
        joined = "；".join(dict.fromkeys(title_cells))
        blocks.append(head + joined[:300])
    for grid in split_grids(rows):
        if len(grid) < THRESH["min_rows"]:
            continue
        layout = classify(grid)
        if layout.kind == FLAT:
            flat = "\n".join("；".join(str(v).strip() for v in r if not is_blank(v)) for r in grid)
            if flat.strip():
                blocks.append(flat)
            warnings.extend(layout.warnings)
            continue
        records = to_records(grid, layout)
        # 转置表的记录标题本身就是主键（型号名），每个块都会带上，无需再复制字段
        keys = [] if layout.kind == TRANSPOSED else None
        blocks.extend(
            render_blocks(records, sheet_name=sheet_name, max_chars=max_chars, key_fields=keys)
        )
        warnings.extend(layout.warnings)
    return blocks, warnings
