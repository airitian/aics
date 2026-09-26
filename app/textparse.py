"""文档解析：把上传文件转成纯文本。

失败必须给**可操作提示**，不许静默失败（PRD 3.2 边界表）。
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from app.config import settings
from app.tablelayout import parse_sheet
from app.textstruct import clean_pdf_pages, parse_docx, parse_markdown, parse_plain_text, render_blocks
from app import vision as _vision


class ParseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class ParseResult:
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    # 解析层给出的**原子块**（表格记录）。非空时入库原样成块，不参与并段；
    # 只有表格类解析会填它，其余格式留空，走 chunk_text 的段落滑窗。
    blocks: list[str] = field(default_factory=list)
    # 通过「整页渲染 + 视觉模型转写」拿到文字的页码（仅 PDF 扫描页兜底会填）。
    # 上层据此**跳过这些页的内嵌图片识别**：扫描页的内嵌图就是整页扫描图，
    # 内容已经抄成文字了，再送一遍视觉模型是双倍开销、产出重复块。
    ocr_pages: set[int] = field(default_factory=set)


SUPPORTED_HINT = "支持的格式：" + "、".join(sorted(settings.allowed_ext))


def _decode_bytes(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
        try:
            return data.decode(enc), warnings
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), ["文件编码无法自动识别，已按替换字符处理"]


def _ocr_pdf_pages(data: bytes, page_nos: list[int]) -> tuple[dict[int, str], list[str]]:
    """把指定页渲染成图并送视觉模型转写（扫描版 PDF 兜底的核心一步）。

    这里是**同步桥**：parse_document 跑在线程池里（没有任何事件循环），
    asyncio.run 在本线程现起现关一个 loop 是安全的 —— 不能在主线程的
    loop 里同步等视觉模型，也不能给 upload 接口再包一层 await。
    视觉调用全部走 vision 模块的属性（describe_many / enabled），
    是为了让测试能按模块打桩，而不是把桩焊死在解析层。
    """
    import asyncio

    from app import docimage as docimage_mod
    from app import vision as vision_mod

    rendered, render_warnings = docimage_mod.render_pdf_pages(
        data, page_nos, dpi=settings.pdf_ocr_dpi
    )
    if not rendered:
        return {}, render_warnings
    items = [(no, blob, mime, "") for no, blob, mime in rendered]
    texts = asyncio.run(vision_mod.describe_many(items, mode="page"))
    return texts, render_warnings


_NO_TEXT_LAYER = "该 PDF 没有可提取的文字层（可能是扫描件或图片版）"
_TEXT_VERSION_HINT = "请提供可复制文本的版本（如导出的文本版 PDF、DOCX 或 TXT）"


def _parse_pdf(data: bytes) -> ParseResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("服务端未安装 PDF 解析组件（pypdf），请联系管理员") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ParseError(f"PDF 无法打开：{exc}") from exc

    # 加密 PDF：先试空密码（很多文件只是设了权限口令，能直接解开）。
    # 关键点：`decrypt()` 解不开时**不抛异常**，只返回 NOT_DECRYPTED ——
    # 若只看异常就会继续往下读 pages，那里才炸出 FileNotDecryptedError，
    # 用户看到的是「处理失败：FileNotDecryptedError」而不是「该 PDF 已加密」。
    # 所以要判返回值（PasswordType：0=没解开，1/2=解开了）。
    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = bool(reader.decrypt(""))
        except Exception:  # noqa: BLE001 - 解密失败一律按未解开处理
            unlocked = False
        if not unlocked:
            raise ParseError(
                "该 PDF 已加密（需要密码），请提供未加密版本，或另存为 DOCX/TXT 后上传"
            )

    try:
        pages = list(reader.pages)
    except Exception as exc:
        raise ParseError(f"PDF 页面无法读取：{exc}") from exc

    page_texts: list[tuple[int, str]] = []
    empty_pages: list[int] = []
    for idx, page in enumerate(pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        if content.strip():
            page_texts.append((idx, content))
        else:
            empty_pages.append(idx)

    # ------------------------------------------------------------------ #
    # 扫描页兜底：没有文字层的页整页渲染成图，送视觉模型逐字转写。
    # 有文字层的页**绝不**再送 OCR —— 文本层比视觉识别准，重做只会引入误差。
    # ------------------------------------------------------------------ #
    ocr_texts: dict[int, str] = {}
    ocr_warnings: list[str] = []
    ocr_pages: set[int] = set()
    overflow: list[int] = []
    if empty_pages:
        capped = set(empty_pages[: settings.pdf_ocr_max_pages])
        overflow = [p for p in empty_pages if p not in capped]
        ocr_texts, ocr_warnings = _ocr_pdf_pages(data, sorted(capped))
        ocr_pages = set(ocr_texts)

    if not page_texts and not ocr_pages:
        # 整本没有文字层、转写又颗粒无收 —— 按原因给**可操作**提示，
        # 不许只甩一句「处理失败」。
        if not _vision.enabled():
            raise ParseError(
                f"{_NO_TEXT_LAYER}。服务端支持扫描页自动识别，"
                "但当前未启用视觉模型（VISION_PROVIDER），请联系管理员启用后重试；"
                f"或{_TEXT_VERSION_HINT}"
            )
        detail = "；".join(ocr_warnings) if ocr_warnings else "视觉模型未能识别出页面文字"
        raise ParseError(f"{_NO_TEXT_LAYER}，且扫描页文字识别失败（{detail}）。{_TEXT_VERSION_HINT}")

    parts: list[str] = []
    if ocr_pages:
        shown = "、".join(str(p) for p in sorted(ocr_pages)[:10])
        more = f" 等 {len(ocr_pages)} 页" if len(ocr_pages) > 10 else ""
        ocr_warnings.insert(
            0,
            f"第 {shown}{more} 页为扫描/图片页，已通过视觉模型转写文字入库（识别可能有误差）",
        )
    if overflow:
        shown = "、".join(str(p) for p in overflow[:10])
        more = f" 等 {len(overflow)} 页" if len(overflow) > 10 else ""
        ocr_warnings.append(
            f"扫描页超过单文档转写上限（{settings.pdf_ocr_max_pages} 页），第 {shown}{more} 未做识别"
        )
    # 文本层页与转写页按页码归位，保持原始阅读顺序
    merged = page_texts + [(p, ocr_texts[p]) for p in ocr_pages]
    merged.sort(key=lambda x: x[0])
    parts = [text for _, text in merged]

    # 剥页眉页脚 + 把被 PDF 切断的行拼回完整句子，再按结构分块。
    # 直接按页拼会留下半截句子，召回时每个块都读不通。
    full, struct_warnings = clean_pdf_pages(parts)
    blocks = render_blocks(parse_plain_text(full), max_chars=settings.chunk_size)
    text = full.strip() or "\n\n".join(parts).strip()
    if not text:
        raise ParseError(f"{_NO_TEXT_LAYER}，{_TEXT_VERSION_HINT}")
    warnings = list(struct_warnings) + list(ocr_warnings)
    still_empty = [p for p in empty_pages if p not in ocr_pages]
    if still_empty:
        # 报具体页码：只说「有 N 页没文字」用户没法判断丢的是什么，
        # 报出页码他才知道要不要补一份文本版。
        shown = "、".join(str(p) for p in still_empty[:10])
        more = f" 等 {len(still_empty)} 页" if len(still_empty) > 10 else ""
        warnings.append(
            f"第 {shown}{more} 页未能提取到文字（可能是扫描图片页），内容未入库"
        )
    return ParseResult(text=text, blocks=blocks, warnings=warnings, ocr_pages=ocr_pages)


def _parse_docx(data: bytes) -> ParseResult:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ParseError("服务端未安装 DOCX 解析组件（python-docx），请联系管理员") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParseError(f"DOCX 无法打开：{exc}") from exc

    struct_blocks, struct_warnings = parse_docx(document, max_chars=settings.chunk_size)
    if not struct_blocks:
        # 结构层没抓到东西（少见），退回原来的「段落 + 表格裸拼」，保证不丢内容
        parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        text = "\n".join(parts).strip()
        if not text:
            raise ParseError("该 DOCX 未提取到任何文字内容，请确认文件内容")
        return ParseResult(text=text, warnings=struct_warnings)

    blocks = render_blocks(struct_blocks, max_chars=settings.chunk_size)
    if not blocks:
        raise ParseError("该 DOCX 未提取到任何文字内容，请确认文件内容")
    return ParseResult(text="\n\n".join(blocks), blocks=blocks, warnings=struct_warnings)


def _parse_text(data: bytes) -> ParseResult:
    text, warnings = _decode_bytes(data)
    text = text.strip()
    if not text:
        raise ParseError("文件内容为空")
    blocks = render_blocks(parse_plain_text(text), max_chars=settings.chunk_size)
    return ParseResult(text=text, blocks=blocks, warnings=warnings)


def _parse_markdown(data: bytes) -> ParseResult:
    text, warnings = _decode_bytes(data)
    if not text.strip():
        raise ParseError("文件内容为空")
    blocks = render_blocks(
        parse_markdown(text, max_chars=settings.chunk_size), max_chars=settings.chunk_size
    )
    return ParseResult(text=text.strip(), blocks=blocks, warnings=warnings)


def _parse_csv(data: bytes) -> ParseResult:
    text, warnings = _decode_bytes(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
        warnings.append("未能识别分隔符，已按逗号处理")

    reader = csv.reader(io.StringIO(text), dialect)
    rows = [r for r in reader]
    if not rows:
        raise ParseError("CSV 内容为空")

    blocks, layout_warnings = parse_sheet(rows, max_chars=settings.chunk_size)
    if not blocks:
        raise ParseError("CSV 只有表头没有数据行")
    return ParseResult(
        text="\n\n".join(blocks),
        blocks=blocks,
        warnings=warnings + layout_warnings,
    )


def _parse_xlsx(data: bytes) -> ParseResult:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise ParseError("服务端未安装 XLSX 解析组件（openpyxl），请联系管理员") from exc

    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise ParseError(f"表格无法打开：{exc}。若为加密或损坏文件，请另存后重试") from exc

    blocks: list[str] = []
    warnings: list[str] = []
    for sheet in wb.worksheets:
        # 只读模式信任文件里声明的 <dimension>；部分导出工具（1688 订单导出实测过）
        # 会写错成 "A1"，openpyxl 于是只返回第 1 行 —— 630 行订单数据会被静默丢掉，
        # 表现是"上传成功、但 AI 怎么问都查不到"。重置后改为一直读到流末尾。
        sheet.reset_dimensions()
        rows = [list(r) for r in sheet.iter_rows(values_only=True)]
        sheet_blocks, sheet_warnings = parse_sheet(
            rows, sheet_name=sheet.title, max_chars=settings.chunk_size
        )
        if not sheet_blocks:
            warnings.append(f"工作表「{sheet.title}」无有效数据行，已跳过")
            continue
        blocks.extend(sheet_blocks)
        for w in sheet_warnings:
            warnings.append(f"工作表「{sheet.title}」：{w}")
    wb.close()

    if not blocks:
        raise ParseError("表格中未提取到任何数据行")
    return ParseResult(text="\n\n".join(blocks), blocks=blocks, warnings=warnings)


def _xls_cell(cell) -> str:
    """xlrd 单元格 → 文本。数字去掉无意义的 .0，日期转 ISO，错误/空返回空串。"""
    import xlrd

    if cell.ctype == xlrd.XL_CELL_TEXT:
        return str(cell.value).strip()
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        return str(int(cell.value)) if float(cell.value).is_integer() else str(cell.value)
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            import datetime as _dt

            return _dt.datetime(
                *xlrd.xldate_as_tuple(cell.value, 0)
            ).isoformat(sep=" ", timespec="seconds")
        except Exception:  # noqa: BLE001 — 个别非法日期值不中断整表
            return str(cell.value)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if cell.value else "FALSE"
    return ""  # EMPTY / BLANK / ERROR


def _parse_xls(data: bytes) -> ParseResult:
    """老版 .xls（BIFF）解析：xlrd 读值 → 转 rows → 复用 xlsx 同一套表格布局逻辑。"""
    try:
        import xlrd
    except ImportError as exc:  # pragma: no cover
        raise ParseError("服务端未安装 XLS 解析组件（xlrd），请联系管理员") from exc

    try:
        wb = xlrd.open_workbook(file_contents=data)
    except Exception as exc:
        raise ParseError(
            f"表格无法打开：{exc}。若为加密或损坏文件，请在 Excel 中另存为 .xlsx 后重试"
        ) from exc

    blocks: list[str] = []
    warnings: list[str] = []
    for sheet in wb.sheets():
        rows: list[list[str]] = []
        for r in range(sheet.nrows):
            cells = [_xls_cell(sheet.cell(r, c)) for c in range(sheet.ncols)]
            # 掐掉行尾连续空格，避免整行空白干扰表头判定
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.append(cells)
        sheet_blocks, sheet_warnings = parse_sheet(
            rows, sheet_name=sheet.name, max_chars=settings.chunk_size
        )
        if not sheet_blocks:
            warnings.append(f"工作表「{sheet.name}」无有效数据行，已跳过")
            continue
        blocks.extend(sheet_blocks)
        for w in sheet_warnings:
            warnings.append(f"工作表「{sheet.name}」：{w}")

    if not blocks:
        raise ParseError("表格中未提取到任何数据行")
    return ParseResult(text="\n\n".join(blocks), blocks=blocks, warnings=warnings)


_DISPATCH = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".txt": _parse_text,
    ".md": _parse_markdown,
    ".csv": _parse_csv,
    ".xlsx": _parse_xlsx,
    ".xls": _parse_xls,
}


def parse_document(filename: str, data: bytes) -> ParseResult:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    parser = _DISPATCH.get(ext)
    if parser is None:
        raise ParseError(f"不支持的格式 {ext or '（无扩展名）'}。{SUPPORTED_HINT}")
    if not data:
        raise ParseError("文件内容为空")
    result = parser(data)
    result.text = _normalize(result.text)
    result.blocks = [b for b in (_normalize(b) for b in result.blocks) if b]
    prefix = (
        _build_overview(filename, result.blocks)
        + _build_attribute_summaries(filename, result.blocks)
        + _build_kv_param_summaries(filename, result.blocks)
        + _build_bullet_summaries(filename, result.blocks)
        + _build_service_summary(filename, result.blocks)
        + _build_first_use_summary(filename, result.blocks)
    )
    prefix = [b for b in prefix if b]
    if prefix:
        result.blocks[:0] = prefix
        result.text = "\n\n".join(prefix) + "\n\n" + result.text
    return result


# 表格记录块的首行形如「【工作表名】型号：字段...」或「【工作表名】Model 名：...」。
_RECORD_HEAD = re.compile(r"^【([^】]+)】([^：:]{1,40})[：:]")

# 记录块正文里的「字段: 值」段（键不含冒号、长度受限，防止把值里的冒号误当字段）。
_RECORD_KV = re.compile(r"^([^：:]{1,40})[：:]\s*(.*)$", re.S)


def _parse_record_blocks(blocks: list[str]) -> list[tuple[str, str, dict[str, str]]]:
    """把 render_blocks 产出的记录块反向解析成 (工作表, 条目名, {字段: 值})。

    为什么不直接传结构化记录：parse_sheet 的三个入口（xlsx/xls/csv）已稳定返回
    (blocks, warnings)，在这里从成品块反解可以不动 tablelayout 的对外契约。
    值本身含「；」时（如阶梯报价「＜80PCS,￥165/PCS；＜240PCS,￥160/PCS」）会被
    拆开，没有「字段:」形态的残段并回上一个字段，保证信息不丢。
    """
    records: list[tuple[str, str, dict[str, str]]] = []
    for b in blocks:
        m = _RECORD_HEAD.match(b)
        if not m:
            continue
        sheet, title = m.group(1).strip(), m.group(2).strip()
        if not title:
            continue
        fields: dict[str, str] = {}
        last_key = ""
        for seg in b[m.end():].split("；"):
            seg = seg.strip()
            if not seg:
                continue
            km = _RECORD_KV.match(seg)
            if km and km.group(1).strip():
                last_key = km.group(1).strip()
                fields[last_key] = km.group(2).strip()
            elif last_key:
                fields[last_key] = f"{fields[last_key]}；{seg}"
        records.append((sheet, title, fields))
    return records


# 常见表格字段的中英对照。知识库多为英文参数表、访客多用中文提问，
# 属性对照块的标题带上中文别名，中文问题（"每个网桥的重量"）才对得上向量。
# 顺序即匹配优先级：长字段名必须排在其子串之前（"ctn weight" 先于 "weight"）。
_FIELD_ALIAS: list[tuple[str, str]] = [
    ("ctn weight", "整箱重量"),
    ("gross weight", "毛重"),
    ("net weight", "净重"),
    ("weight", "重量"),
    ("wireless data rate", "无线速率"),
    ("data rate", "速率"),
    ("transmission distance", "传输距离"),
    ("ptp distance", "传输距离"),
    ("distance", "距离"),
    ("frequency", "频率"),
    ("antenna", "天线"),
    ("chipset", "芯片"),
    ("flash/ram", "内存"),
    ("lan ports", "网口"),
    ("poe ports", "POE口"),
    ("ports", "端口"),
    ("poe power", "供电"),
    ("power", "功率"),
    ("ctn size", "整箱尺寸"),
    ("packing", "装箱"),
    ("package", "包装"),
    ("accessory", "配件"),
    ("warranty", "保修"),
    ("price", "价格"),
    ("moq", "最小起订量"),
    ("color", "颜色"),
    ("material", "材质"),
    ("model", "型号"),
]

# 概览「条目速览」行优先挑的字段类别（按区分度排序，最多取 3 个）。
_OVERVIEW_PREF = ("频率", "距离", "速率", "天线", "重量", "价格")


def _field_alias(field: str) -> str:
    fl = field.lower().strip()
    for en, cn in _FIELD_ALIAS:
        if en in fl:
            return cn
    # 中文表头直接当别名（"价格"/"重量"这类字段名访客本来就会问）
    if fl and any("\u4e00" <= ch <= "\u9fff" for ch in fl) and len(fl) <= 8:
        return fl
    return ""


def _pick_overview_attrs(fields: dict[str, str], limit: int = 3) -> list[str]:
    """从一条记录里挑出最能区分型号的少数几个字段值，做概览速览行。"""
    picked: list[str] = []
    used: set[str] = set()
    for field, value in fields.items():
        if len(picked) >= limit:
            break
        if len(value) > 20:
            continue
        alias = _field_alias(field)
        if not alias or alias in used:
            continue
        for pref in _OVERVIEW_PREF:
            if pref in alias or alias in pref:
                used.add(alias)
                picked.append(value)
                break
    return picked


def _build_overview(filename: str, blocks: list[str]) -> list[str]:
    """给表格类文档生成「文档概览」块（+「条目速览」块），放在 blocks 最前。

    为什么需要：枚举型问题（"你有哪些网桥/都有什么型号"）只召回 top_k 条记录，
    模型只能列出被召回的那几款，用户会以为答案不完整。概览块把全部条目名压成
    一条可检索的清单，这类问题就能整体命中。

    条目速览单独成块（每款带 2-3 个关键参数值）：检索只看块文本，有了速览块，
    "按速率/距离分组介绍全部型号"这类追问也能一次命中，而不是只报一串光秃秃
    的型号名。两块分开还能避免大表把概览撑过 chunk_size 被滑窗切半。

    只对表格记录块生效（首行匹配 【x】y：）；正文类文档的块是段落/标题结构，
    造清单反而稀释语义。
    """
    names: list[str] = []
    attrs: dict[str, list[str]] = {}
    seen: set[str] = set()
    sheets: set[str] = set()
    for sheet, title, fields in _parse_record_blocks(blocks):
        sheets.add(sheet)
        if title.lower() in seen:
            continue
        seen.add(title.lower())
        names.append(title)
        picked = _pick_overview_attrs(fields)
        if picked:
            attrs[title] = picked
    if len(names) < 5:
        return []  # 条目太少没有枚举价值，正常检索就能覆盖
    # 单工作表时前缀纯冗余（"Wireless bridge-WB730" x17 → "WB730、WB620E…"）
    if len(sheets) == 1:
        sheet_note = f"（工作表「{next(iter(sheets))}」）"
    else:
        sheet_note = ""
    joined = "、".join(names)
    if len(joined) > 500:
        joined = joined[:500] + "…"
    out = [f"【文档概览】《{filename}》{sheet_note}共包含 {len(names)} 个条目：{joined}"]
    # 条目很多时速览行会把块撑得过长（检索窗口切开反而丢信息），只保留纯清单
    if len(names) <= 40:
        detail = "；".join(
            f"{name}: {'/'.join(attrs[name])}" for name in names if attrs.get(name)
        )
        if detail:
            out.append(f"【条目速览】《{filename}》各条目关键参数：{detail}")
    return out


# 属性对照块里，字段值平均长度超过此值的不做对照（是描述性文字，不是参数）。
_ATTR_MAX_AVG_LEN = 40


def _build_attribute_summaries(filename: str, blocks: list[str]) -> list[str]:
    """把表格记录按「字段」转置聚合，每个字段生成一条「属性对照」块。

    为什么需要：参数对照表按条目（型号）竖切成块后，"每个网桥对应的重量是多少"
    这类**横向对照**问题只能召回零星几个型号的块，答案必然残缺。
    转置后「CTN weight：WB730=15.5KG；WB620E=13KG；…」聚成一条，
    一次召回就是完整答案——这是多粒度索引的常规做法（条目块 + 属性块冗余并存）。

    生成规则（三者同时满足才生成，避免制造噪声块）：
    - 覆盖条目数 ≥5（与概览块同一门槛，少条目正常检索就能覆盖）；
    - 字段值平均长度 ≤40 字符（长文本字段是描述不是参数，聚合只会稀释语义）；
    - 字段值不全相同（全员同值的字段没有逐条对照的价值，任何单块里都有）。
    """
    records = _parse_record_blocks(blocks)
    if len(records) < 5:
        return []

    # 按字段聚合；同一条目同字段出现多次时（超长记录拆块会复制主键字段）
    # 后值覆盖前值——同一型号的参数在不同块里理应一致，重复只来自拆块复制。
    agg: dict[str, dict[str, str]] = {}
    for _sheet, title, fields in records:
        for field, value in fields.items():
            if value:
                agg.setdefault(field, {})[title] = value

    out: list[str] = []
    for field, by_title in agg.items():
        pairs = list(by_title.items())
        if len(pairs) < 5:
            continue
        avg_len = sum(len(v) for _, v in pairs) / len(pairs)
        if avg_len > _ATTR_MAX_AVG_LEN:
            continue
        if len({v for _, v in pairs}) <= 1:
            continue
        alias = _field_alias(field)
        label = f"{field}（{alias}）" if alias and alias != field else field
        header = f"【属性对照】《{filename}》{label}："
        # 按 chunk_size 打包；超长时续块沿用同一前缀，保证每块自足可检索
        pieces: list[str] = []
        cur: list[str] = []
        cur_len = len(header)
        for title, value in pairs:
            item = f"{title}: {value}；"
            if cur and cur_len + len(item) > settings.chunk_size:
                pieces.append(header + "".join(cur).rstrip("；"))
                cur, cur_len = [], len(header)
            cur.append(item)
            cur_len += len(item)
        if cur:
            pieces.append(header + "".join(cur).rstrip("；"))
        # 全部塞进一块还超长的字段（条目极多时）天然由入库层滑窗兜底
        out.extend(pieces)
    return out


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- 纯文本参数区（说明书类 PDF 的「键 值」行） ----------
# 为什么需要：说明书 PDF 的产品参数/保养周期区是纯文本行（"产品型号 DDX19 / DDX29"），
# 不是表格记录块，上方的属性对照机制（_parse_record_blocks）覆盖不到。这些行散在
# 长段落块里，「这个扫地机器人的型号是什么」这类口语化提问与长块词面重叠极低
# （实测 0.28 分，远低于 0.45 召回门槛），模型只能答"没查到"。把键值行抽成一条
# 致密的【参数对照】块，与提问词面直接对齐，一次召回即含全部参数。
#
# 噪声防线（三层，全过才收）：
# 1. 键：2-14 字、无空白、不含句读标点、含中文（或 ≥3 位纯英文词），排除页码/序号；
# 2. 值：1-40 字、不含句读标点——参数值是短符号串，散文句几乎必然带句号/逗号；
# 3. 密度：只有周边 ±5 行内至少还有 2 个同类键值行（成片参数区）才收录，
#    孤立的零星匹配（正文偶合）一律丢弃。

_KV_KEY_CHARS = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9（）()\-－/／·～~．.]+$")
_KV_VALUE_BAD = re.compile(r"[。！？；，、：:]")
_KV_BULLET = re.compile(r"^[•·\-–—*]\s*")


def _kv_key_ok(key: str) -> bool:
    if not (2 <= len(key) <= 14) or not _KV_KEY_CHARS.match(key):
        return False
    if key.isdigit():
        return False  # 页码 / 序号
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in key)
    if has_cjk:
        return True
    return len(key) >= 3  # 纯英文键（Model/Power）至少 3 字符，防单字母噪声


def _kv_split_line(line: str) -> tuple[str, str] | None:
    """「键 值」行 → (键, 值)；纯键行 / 不合格行 → None。"""
    line = _KV_BULLET.sub("", line.strip())
    if not line:
        return None
    parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    key, value = parts[0].strip(), parts[1].strip()
    if not _kv_key_ok(key):
        return None
    if not value or len(value) > 40 or _KV_VALUE_BAD.search(value):
        return None
    return key, value


def _kv_dense_pairs(lines: list[str]) -> list[list[tuple[str, str]]]:
    """抽一个块里的键值行，按「连续区域」分段返回。

    每段是同一参数区的键值对（如产品参数页、保养周期表）；段与段之间以
    连续 ≥3 行非键值行（散文/图注）为界。分段是关键：产品参数块和保养
    周期块混在一起会互相稀释语义，「型号」类提问召回时两头不讨好——
    分开后每块主题单一，检索各归各的提问。

    密度过滤：每个命中行在 ±5 行窗口内至少还有 2 个同类命中才算数。
    """
    found: list[tuple[int, tuple[str, str]]] = []
    i, n = 0, len(lines)
    while i < n:
        parsed = _kv_split_line(lines[i])
        if parsed:
            found.append((i, parsed))
            i += 1
            continue
        # 跨行断键：PDF 会把长键折行（"额定输入电流（充电状" / "态） 0.5A"），
        # 本行只有键、下一行是「键尾 值」→ 拼回一个键。
        tok = _KV_BULLET.sub("", lines[i].strip())
        if i + 1 < n and 2 <= len(tok) <= 14 and _kv_key_ok(tok):
            nxt = _kv_split_line(lines[i + 1])
            if nxt:
                merged_key = tok + nxt[0]
                if len(merged_key) <= 16 and _kv_key_ok(merged_key):
                    found.append((i, (merged_key, nxt[1])))
                    i += 2
                    continue
        i += 1

    # 密度过滤：每个命中行在 ±5 行窗口内至少还有 2 个同类命中
    idxs = [i for i, _ in found]
    kept: list[tuple[int, tuple[str, str]]] = []
    for i, pair in found:
        neighbors = sum(1 for j in idxs if j != i and abs(j - i) <= 5)
        if neighbors >= 2:
            kept.append((i, pair))

    # 按行号间隙分段：间隔 >3 行视为不同参数区
    segments: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    prev_line: int | None = None
    for i, pair in kept:
        if prev_line is not None and i - prev_line > 3:
            if cur:
                segments.append(cur)
            cur = []
        cur.append(pair)
        prev_line = i
    if cur:
        segments.append(cur)
    return segments


# 客服/售后热线号（常见于说明书散文句："请联系我们的售后服务中心： 400-886-8888"）。
# 散文句带标点进不了键值行抽取，但它恰是"机器坏了找谁"这类高频提问的答案，
# 值得专项提取后并入参数块。
_KV_HOTLINE_RE = re.compile(
    r"(?:售后|服务|客服)(?:服务)?(?:热线|电话|中心)[：:]?\s*"
    r"(4\d{2}[-\s]?\d{3}[-\s]?\d{4}|\d{3,4}[-\s]?\d{7,8})"
)


def _build_kv_param_summaries(filename: str, blocks: list[str]) -> list[str]:
    """把纯文本参数区按连续区域聚成【参数对照】块（与【属性对照】同为多粒度冗余索引）。

    门槛：每个连续区域抽满 5 对才生成——参数区太小（1-4 对）正常检索就能覆盖，
    生成块反而稀释语义。键跨区域重复时（滑窗切块会复制）保留首见。
    售后热线/官网等联系方式不在这里处理（散进密集参数块只会稀释语义），
    由 _build_service_summary 专项成块。
    """
    out: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        for segment in _kv_dense_pairs(block.split("\n")):
            pairs = [(k, v) for k, v in segment if k not in seen]
            seen.update(k for k, _ in pairs)
            if len(pairs) < 5:
                continue
            out.extend(_pack_kv_block(filename, pairs))
    return [b for b in out if b]


def _pack_kv_block(filename: str, pairs: list[tuple[str, str]]) -> list[str]:
    """把一批键值对打包成【参数对照】块（超 chunk_size 时续块沿用同一前缀）。"""
    header = f"【参数对照】《{filename}》"
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = len(header)
    for key, value in pairs:
        item = f"{key}：{value}；"
        if cur and cur_len + len(item) > settings.chunk_size:
            pieces.append(header + "".join(cur).rstrip("；"))
            cur, cur_len = [], len(header)
        cur.append(item)
        cur_len += len(item)
    if cur:
        pieces.append(header + "".join(cur).rstrip("；"))
    return pieces


# ---------- 散文要点区（说明书 PDF 的 bullet 密集区） ----------
# 说明书的使用要求/注意事项常以「•」短句成片出现（摆放要求、清洁剂使用、
# 网络环境等）。这些行混在长段落块里时语义被严重稀释——实测：全能基站的
# 摆放要点与 App 隐私政策同块，「基站能放床头吗」召回的全是隐私政策碎片。
# 把 bullet 密集区抽成独立的【要点清单】块，并吸收紧邻的要求句（如
# "靠墙放置…左右0.1m…前方0.8m"这类不带 bullet 标记的正文），
# 一次召回即含完整要求。
_REQ_KEYWORD_RE = re.compile(r"请|勿|不要|建议|需|禁止|必须|确保|切勿|应")
_STEP_HEAD_RE = re.compile(r"^\d+\s*[\.、．]")  # "2. 寻找合适的…"这类步骤标题
_BULLET_END = ("。", "！", "？", "；")
_BULLET_MIN_LEN = 12     # 短于此的 bullet 行是图注/标签噪声（"语音仅X5 PRO标配"）
_SEG_MIN_BULLETS = 2     # 一个连续区域至少 2 条 bullet 才成块（"存放/充电注意"常只有成对两条）
_CTX_LINES = 3           # 向段前后各看几行正文（吸收紧邻要求句）

# 原子块的模式门控：只给"条件 + 周期/场景"类长尾答案句单独成块
# （如"储存前请先充满电…每 1.5 个月补充电""待机约5h进入深度休眠"）。
_ATOMIC_HINT_RE = re.compile(
    r"每\s*\d+(?:\.\d+)?\s*(?:个)?月|每周|每月|长期|存放|储存|休眠|唤醒|待机|补充电"
)


def _bullet_candidates(lines: list[str]) -> list[tuple[int, str, bool]]:
    """抽出一个块里的 bullet 行 → [(行号, 完整句, 是否编号行)]；折行拼回完整句。

    除圆点/短横 bullet 外，编号步骤行（"5. 返回全能基站"）同样算候选——
    说明书大量用编号列表承载关键操作要求，其正文常折行跟在编号后。
    """
    out: list[tuple[int, str, bool]] = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i].strip()
        stripped = _KV_BULLET.sub("", raw)
        is_numbered = bool(stripped) and _STEP_HEAD_RE.match(stripped) is not None
        if is_numbered:
            stripped = _STEP_HEAD_RE.sub("", stripped).strip()
        if not stripped:  # 必须带 bullet 标记或编号
            i += 1
            continue
        if not is_numbered and stripped == raw:  # 无标记的非编号行跳过
            i += 1
            continue
        text = stripped
        j = i
        extra = 0
        # PDF 折行：句末无标点的 bullet 与后续行拼回（"…推荐按" + "照 1:200 …"）。
        # 最多拼 3 行且不跨空行——空行几乎必然是话题边界，跨过去会把下一节的
        # 句子拼进本条（"…关闭集尘仓" 吞进 "请注意，主机无法在电源关闭…"）；
        # 拼完必须落在句末标点上，半截句直接丢弃。
        while (
            j + 1 < n
            and extra < 3
            and len(text) < 100
            and not text.endswith(_BULLET_END)
            and lines[j + 1].strip()
        ):
            j += 1
            extra += 1
            text += _KV_BULLET.sub("", lines[j].strip())
        if _BULLET_MIN_LEN <= len(text) <= 100 and (j == i or text.endswith(_BULLET_END)):
            out.append((i, text, is_numbered))
            i = j + 1
        else:
            i += 1
    return out


def _bullet_dense_segments(
    lines: list[str],
) -> list[list[tuple[int, str, bool]]]:
    """bullet 密集区分段：±5 行内至少 2 条 bullet 才收录，间隙 >3 行分段。

    密度过滤同 _kv_dense_pairs 的思路——散落的孤立 bullet（图注偶合）不成块。
    门槛取 2 而非 3：说明书"存放/充电注意"一类小节常只有成对两条 bullet，
    卡 3 会把整节连同紧邻的关键要求句一起丢掉（实测教训）。
    """
    cands = _bullet_candidates(lines)
    idxs = [i for i, _, _ in cands]
    kept = [
        (i, t, num)
        for i, t, num in cands
        if sum(1 for j in idxs if j != i and abs(j - i) <= 5) >= 1
    ]
    segments: list[list[tuple[int, str, bool]]] = []
    cur: list[tuple[int, str, bool]] = []
    prev: int | None = None
    for i, t, num in kept:
        if prev is not None and i - prev > 3:
            if cur:
                segments.append(cur)
            cur = []
        cur.append((i, t, num))
        prev = i
    if cur:
        segments.append(cur)
    return segments


def _bullet_context(lines: list[str], lo: int, hi: int) -> list[tuple[int, str]]:
    """吸收 bullet 段前后紧邻的「要求句」（如靠墙放置+间距那句话不带 bullet）。

    只收：段外 ±3 行内、非 bullet 行、非步骤标题、折行拼到句末（最多拼 3 行）
    后 12-90 字、含要求关键词的完整句。间距标注行（"≥0.1m (0.33')"）无关键词
    自然被排除。
    """
    picked: list[tuple[int, str]] = []
    n = len(lines)
    for i in range(max(0, lo - _CTX_LINES), min(n, hi + _CTX_LINES + 1)):
        raw = lines[i].strip()
        if not raw or _KV_BULLET.sub("", raw) != raw:
            continue
        if _STEP_HEAD_RE.match(raw):
            continue
        j, merged = i, raw
        # 折行拼到句末为止（"…范围内不要" + "放置物品。"），拼不满整句就丢弃
        while j + 1 < n and len(merged) < 90 and not merged.endswith(_BULLET_END):
            j += 1
            merged += lines[j].strip()
        merged = _KV_BULLET.sub("", merged).strip()
        if not merged.endswith(_BULLET_END):
            continue
        if 12 <= len(merged) <= 90 and _REQ_KEYWORD_RE.search(merged):
            picked.append((i, merged))
    return picked


def _norm_key(t: str) -> str:
    """句子归一化：去步骤号/提示前缀/空白，用于块内近重复判断。"""
    t = _STEP_HEAD_RE.sub("", t.strip())
    t = re.sub(r"^(温馨提示|注意|警告|提示)[：:]?", "", t)
    return re.sub(r"\s+", "", t)


def _build_bullet_summaries(filename: str, blocks: list[str]) -> list[str]:
    """把 bullet 密集区聚成【要点清单】块（多粒度冗余索引之一）。"""
    out: list[str] = []
    seen_blocks: set[str] = set()
    atomic_norms: list[str] = []  # 原子句文档级归一化缓存（含跨段折行变体去重）
    for block in blocks:
        lines = block.split("\n")
        for seg in _bullet_dense_segments(lines):
            if len(seg) < _SEG_MIN_BULLETS:
                continue
            lo, hi = seg[0][0], seg[-1][0]
            items: list[tuple[int, str, bool]] = list(seg)
            seg_texts = {t for _, t, _ in seg}
            ctx_texts: list[str] = []
            for i, t in _bullet_context(lines, lo, hi):
                if t not in seg_texts:
                    items.append((i, t, False))
                    ctx_texts.append(t)
            # 包含关系去重：编号候选折行后常与上下文吸收句是同一句的
            # 长短变体（"3秒为保护电池…" / "为保护电池…"），按归一化后
            # 互相包含只保留最长的一条，再按行号还原顺序。
            arr = sorted(
                ((i, t, _norm_key(t), num) for i, t, num in items),
                key=lambda x: len(x[2]),
                reverse=True,
            )
            kept_norms: list[str] = []
            kept_items: list[tuple[int, str, bool]] = []
            for i, t, nt, num in arr:
                if len(nt) >= 10 and any(nt in kn for kn in kept_norms):
                    continue
                kept_norms.append(nt)
                kept_items.append((i, t, num))
            items = sorted(kept_items, key=lambda x: x[0])
            if not items:
                continue
            body = "；".join(
                t if t.endswith(_BULLET_END) else t + "。" for _, t, _ in items
            )
            block_text = f"【要点清单】《{filename}》{body}"
            if block_text not in seen_blocks:
                seen_blocks.add(block_text)
                out.append(block_text)
            # 原子块：只给"高价值散句"单独成块——上下文吸收的要求句（无 bullet
            # 保护的散文金句）+ 编号候选（如"5. 返回全能基站"把深度休眠说明
            # 折进一条）。普通 bullet 不原子化。且仅保留**周期性维护/存放/休眠
            # 场景**的句子：这类"条件+周期"句是说明书问答的长尾答案，且最怕
            # 被同段其他子题稀释（实测打包 0.001 vs 单句 0.35）；全量原子化
            # （146 块）会挤占召回池，把参数表/FAQ 块挤出候选，反而劣化其他问题。
            atomic_sources = list(ctx_texts)
            atomic_sources.extend(t for _, t, num in items if num)
            for t in atomic_sources:
                for s in re.split(r"(?<=[。！？])", t):
                    s = s.strip()
                    if len(s) < _BULLET_MIN_LEN or not _ATOMIC_HINT_RE.search(s):
                        continue
                    if not s.endswith(_BULLET_END):
                        s += "。"
                    ns = _norm_key(s)
                    # 文档级包含去重：折行变体常落进相邻分段，逐段去重管不到
                    if len(ns) >= 10 and any(ns in kn or kn in ns for kn in atomic_norms):
                        continue
                    atomic_norms.append(ns)
                    atomic = f"【要点清单】《{filename}》{s}"
                    if atomic not in seen_blocks:
                        seen_blocks.add(atomic)
                        out.append(atomic)
    return out


# ---------- 售后联系方式（散在散文句里的热线/官网/在线客服） ----------
# "机器坏了要找谁？给个电话"是说明书场景的高频提问，但答案常是带标点的
# 散文句（"请联系我们的售后服务中心： 400-886-8888"），进不了键值行抽取；
# 并进密集参数块又会稀释语义（参数块的向量被型号/电压 dominate，联系方式
# 类提问召回不到）。专项抽成一条小【售后服务】块，与提问词面直接对齐。
_SERVICE_URL_RE = re.compile(r"www\.[A-Za-z0-9.\-]+")


def _build_service_summary(filename: str, blocks: list[str]) -> list[str]:
    """抽取售后服务热线 / 官网 / 微信在线客服，聚成一条【售后服务】块。

    三类信息各自的匹配都很精确（热线号形如 400-xxx-xxxx、URL 前置"官网"、
    "微信"+"在线客服"同句），所以允许全文扫描而不限邻近窗口。
    """
    hotline = website = wechat = None
    for block in blocks:
        for line in block.split("\n"):
            s = line.strip()
            if hotline is None and (m := _KV_HOTLINE_RE.search(s)):
                label = re.split(r"[：:]", m.group(0))[0].strip()
                num = re.sub(r"\s+", "", m.group(1))
                hotline = f"{label}：{num}"
            if website is None and "官网" in s and (m := _SERVICE_URL_RE.search(s)):
                website = "官网：" + m.group(0).rstrip(".。，；")
            if wechat is None and "微信" in s and "在线客服" in s:
                wechat = s
        if hotline and website and wechat:
            break
    if not (hotline or website or wechat):
        return []
    parts = [p for p in (hotline, website, wechat) if p]
    return [f"【售后服务】《{filename}》" + "；".join(parts)]


# ---------- 首次使用准备要点（散在版式混乱块里的"使用前"准备句） ----------
# "刚买回来第一次用要注意什么"是高频提问，但 PDF 双栏解析后「使用前注意」
# 的准备句（收纳低矮物、开门建图、防护栏防跌落、流苏边处理）常与传感器
# 介绍等内容混在同一块，块向量被无关内容 dominate，首次使用类提问召回不到。
# 准备句词面直接对齐提问（"首次使用""收纳""防护栏"），值得专项成块。
_FIRST_USE_HINT_RE = re.compile(r"首次使用|使用前|防护栏|流苏边|收纳好|整理出[^。]*清扫空间")
# 这类门控词命中时句首往往是 PDF 图注杂质（如"10.传感器介绍9.自动上下水模块…"），
# 从门控词处截头，保住句子的语义纯度；其余门控（防护栏/流苏边等）句首可能带
# 有效限定语（"如家中地毯有"），保持原句。
_FIRST_USE_TRIM_RE = re.compile(r"使用前|首次使用|收纳好|整理出")
_SENT_SPLIT_RE = re.compile(r"(?<=[。；])")


def _build_first_use_summary(filename: str, blocks: list[str]) -> list[str]:
    """抽取「首次使用/使用前」准备句，聚成一条【首次使用要点】块。

    门控：句子必须命中 _FIRST_USE_HINT_RE（词面与首次使用强相关），
    且长度 ≤120 字；凑不满 2 句不成块（单句没有稀释问题，不值得占位）。
    """
    sentences: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        if block.startswith("【"):
            continue  # 跳过摘要块，防止自我引用
        for raw in _SENT_SPLIT_RE.split(block):
            s = re.sub(r"\s+", "", raw)
            if not (12 <= len(s) <= 120) or s in seen:
                continue
            m = _FIRST_USE_HINT_RE.search(s)
            if not m:
                continue
            if (t := _FIRST_USE_TRIM_RE.search(s)) and t.start() > 0:
                s = s[t.start():]
                if len(s) < 12:
                    continue
            seen.add(raw)
            sentences.append(s)
    if len(sentences) < 2:
        return []
    return [f"【首次使用要点】《{filename}》" + "；".join(sentences[:8])]
