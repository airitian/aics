"""文档解析（app/textparse）离线回归测试。

为什么单独测这一层：解析失败是**用户唯一能直接看到**的知识库错误，
而它又最容易被后续改动悄悄改坏（改白名单、换解析库、动异常类型）。
这里不碰网络、不碰数据库，只验证「哪些格式能解析、解析出什么、失败时说什么」。
"""
from __future__ import annotations

import io

import pytest

from app.textparse import ParseError, parse_document

# --------------------------------------------------------------------------- #
# 在内存里造各种文件（不依赖 reportlab / fpdf）
# --------------------------------------------------------------------------- #
def _pdf(objects: list[bytes]) -> bytes:
    """拼一个最小合法 PDF（含正确 xref 偏移）。"""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(n).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


def _ascii_pdf(lines: list[str]) -> bytes:
    """有文字层的普通 ASCII PDF。"""
    ops = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
    for i, ln in enumerate(lines):
        esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i:
            ops.append("T*")
        ops.append(f"({esc}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _cjk_pdf(lines: list[str]) -> bytes:
    """中文 PDF：Type0 + Identity-H + ToUnicode（Word / WPS 导出 PDF 用的就是这套）。"""
    chars = sorted({ch for ln in lines for ch in ln})
    cmap = {ch: i + 1 for i, ch in enumerate(chars)}

    ops = ["BT", "/F1 12 Tf", "18 TL", "56 720 Td"]
    for i, ln in enumerate(lines):
        if i:
            ops.append("T*")
        ops.append("<" + "".join(f"{cmap[ch]:04X}" for ch in ln) + "> Tj")
    ops.append("ET")
    content = "\n".join(ops).encode("latin-1")

    bfchars = "\n".join(f"<{cid:04X}> <{ord(ch):04X}>" for ch, cid in cmap.items())
    tounicode = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{len(cmap)} beginbfchar\n{bfchars}\nendbfchar\n"
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode("latin-1")

    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type0 /BaseFont /NotoSansSC /Encoding /Identity-H "
            b"/DescendantFonts [6 0 R] /ToUnicode 8 0 R >>",
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /NotoSansSC "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            b"/FontDescriptor 7 0 R /DW 1000 >>",
            b"<< /Type /FontDescriptor /FontName /NotoSansSC /Flags 4 "
            b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 880 /Descent -200 "
            b"/CapHeight 700 /StemV 80 >>",
            b"<< /Length " + str(len(tounicode)).encode() + b" >>\nstream\n" + tounicode + b"\nendstream",
        ]
    )


def _empty_pdf() -> bytes:
    """有页面但没有任何文字绘制指令 —— 等价于扫描件/纯图片页。"""
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            b"<< /Length 0 >>\nstream\nendstream",
        ]
    )


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def test_pdf_with_text_layer_extracts_ascii():
    text = parse_document("policy.pdf", _ascii_pdf(["Tent Warranty 36 months", "Return within 14 days"])).text
    assert "Tent Warranty 36 months" in text
    assert "Return within 14 days" in text


def test_pdf_with_text_layer_extracts_chinese():
    """中文 PDF 是客服知识库最常见的来源，必须真的能提出中文（而不是乱码）。"""
    src = ["露营帐篷保修与退换政策", "保修：提供36个月厂家保修，覆盖缝线开裂与拉链破损。"]
    text = parse_document("帐篷.pdf", _cjk_pdf(src)).text
    for probe in ("露营帐篷", "36个月", "拉链破损"):
        assert probe in text, f"未提取到 {probe}，实际：{text[:120]!r}"


def test_scanned_pdf_gives_actionable_error():
    """无文字层（扫描件）必须明确告知，并给出可执行的替代方案。"""
    with pytest.raises(ParseError) as ei:
        parse_document("scan.pdf", _empty_pdf())
    msg = str(ei.value)
    assert "文字层" in msg or "扫描" in msg
    assert "DOCX" in msg or "TXT" in msg, "只说失败不够，要告诉用户下一步怎么做"


def test_broken_pdf_fails_cleanly():
    """损坏的 PDF 要抛 ParseError（上层标记该文件失败），不能冒泡成 500。"""
    with pytest.raises(ParseError):
        parse_document("broken.pdf", b"%PDF-1.4 this is not a real pdf")


def test_encrypted_pdf_is_reported():
    """加密 PDF 要么空密码解开、要么给出可操作提示，不能静默产出空文本。"""
    try:
        from pypdf import PdfWriter
    except ImportError:  # pragma: no cover
        pytest.skip("pypdf 未安装")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret-password")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ParseError) as ei:
        parse_document("locked.pdf", buf.getvalue())
    assert "加密" in str(ei.value)


# --------------------------------------------------------------------------- #
# 老版 .xls（BIFF）
# --------------------------------------------------------------------------- #
def _make_xls(rows: list[list], dates: dict[tuple[int, int], str] | None = None) -> bytes:
    """用 xlwt 造一个真实 .xls 文件。dates: {(row,col): 'D-MMM-YY' 格式串} 标记日期单元格。"""
    xlwt = pytest.importorskip("xlwt")
    wb = xlwt.Workbook()
    sheet = wb.add_sheet("产品表")
    date_style = xlwt.easyxf(num_format_str="YYYY-MM-DD")
    dates = dates or {}
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if (r, c) in dates:
                    import datetime as _dt

                    sheet.write(r, c, _dt.datetime.strptime(val, dates[(r, c)]), date_style)
                else:
                    sheet.write(r, c, val)
            else:
                sheet.write(r, c, val)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xls_parsed_with_layout_logic():
    """行式表走行记录拼装；数字不带 .0 尾巴。"""
    data = _make_xls(
        [
            ["型号", "速率", "价格"],
            ["AP-101", "300M", 129],
            ["AP-202", "1200M", 259.5],
        ]
    )
    result = parse_document("设备清单.xls", data)
    joined = "\n".join(result.blocks)
    assert "型号" in joined and "AP-101" in joined and "129" in joined
    assert "129.0" not in joined and "259.50" not in joined.replace("259.5", "")


def test_xls_date_cell_becomes_iso_text():
    import datetime as _dt

    data = _make_xls(
        [["型号", "生产日期"], ["AP-101", "2026-01-15"]],
        dates={(1, 1): "%Y-%m-%d"},
    )
    joined = "\n".join(parse_document("排产.xls", data).blocks)
    assert _dt.datetime(2026, 1, 15).isoformat(sep=" ")[:10] in joined


def test_xls_header_only_sheet_raises_like_xlsx():
    """只有表头没有数据行：与 xlsx 路径同一约定 —— 报错提示，不静默入库。"""
    data = _make_xls([["列A", "列B"]])
    with pytest.raises(ParseError) as ei:
        parse_document("只有表头.xls", data)
    assert "未提取到" in str(ei.value)


def test_corrupted_xls_gives_clear_error():
    with pytest.raises(ParseError) as ei:
        parse_document("old.xls", b"\xd0\xcf\x11\xe0not really a biff file")
    assert "无法打开" in str(ei.value)


# --------------------------------------------------------------------------- #
# 文档概览块（枚举型问题的答案完整性）
# --------------------------------------------------------------------------- #
def test_tabular_doc_gets_overview_block_listing_all_records():
    """表格 ≥5 条记录时必须在最前生成概览块，枚举问题才可能答全。"""
    rows = [["型号", "价格"]] + [[f"WB-{i}", f"{100 + i}元"] for i in range(1, 9)]
    data = _make_xls(rows)
    result = parse_document("型号表.xls", data)
    head = result.blocks[0]
    assert head.startswith("【文档概览】")
    assert "共包含 8 个条目" in head
    for i in range(1, 9):
        assert f"WB-{i}" in head, f"概览必须包含 WB-{i}"


def test_overview_omitted_when_records_too_few():
    data = _make_xls([["型号", "价格"], ["A", "1元"], ["B", "2元"], ["C", "3元"]])
    result = parse_document("小表.xls", data)
    assert not any(b.startswith("【文档概览】") for b in result.blocks)


# --------------------------------------------------------------------------- #
# 属性对照块（横向对照问题的答案完整性）
# --------------------------------------------------------------------------- #
def _bridge_xlsx(models: int = 6) -> bytes:
    """转置参数表：列 = 型号，行 = 字段。模仿真实无线网桥报价表。"""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Wireless bridge"
    ws.append(["参数"] + [f"WB{i}00" for i in range(1, models + 1)])
    ws.append(["Frequency", *[f"5.8GHz" for _ in range(models)]])
    ws.append(["CTN weight", *[f"{10 + i / 2}KG" for i in range(models)]])
    ws.append(["Packing", *[f"{8 + i % 2 * 2} Pairs in CTN" for i in range(models)]])
    ws.append(
        ["PRICE(RMB)", *[f"＜80PCS,￥{100 + i}/PCS；＜240PCS,￥{95 + i}/PCS" for i in range(models)]]
    )
    ws.append(["Accessory", *["24V PoE power, User Manual" for _ in range(models)]])
    ws.append(
        ["Funtional advantage", *["1、IP address management " + "x" * 120 for _ in range(models)]]
    )
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_attribute_summary_block_covers_every_model():
    """「每个型号的重量」这类横向对照问题必须能一次召回全型号答案。"""
    res = parse_document("bridge.xlsx", _bridge_xlsx())
    weight_blocks = [
        b for b in res.blocks if b.startswith("【属性对照】") and "CTN weight" in b
    ]
    assert len(weight_blocks) == 1, f"CTN weight 应恰好聚成一块，实际 {len(weight_blocks)} 块"
    block = weight_blocks[0]
    assert "整箱重量" in block, "标题要带中文别名，中文提问才对得上向量"
    for i in range(1, 7):
        assert f"WB{i}00" in block, f"对照块必须覆盖 WB{i}00"
    assert "10.5KG" in block and "12.5KG" in block


def test_attribute_summary_transposes_packing_and_price():
    res = parse_document("bridge.xlsx", _bridge_xlsx())
    joined = "\n".join(res.blocks)
    assert "Packing（装箱）" in joined
    assert "PRICE(RMB)（价格）" in joined
    # 阶梯报价值内部的「；」不许把值拆碎：同一型号的两档价格要连在一起
    price_block = next(b for b in res.blocks if "PRICE(RMB)" in b)
    assert "＜240PCS,￥96/PCS" in price_block


def test_attribute_summary_skips_uniform_and_prose_fields():
    """全员同值的字段（配件）和长文本字段（功能描述）不做对照，避免噪声块。"""
    res = parse_document("bridge.xlsx", _bridge_xlsx())
    joined = "\n".join(res.blocks)
    for b in res.blocks:
        if b.startswith("【属性对照】"):
            assert "Accessory" not in b, "全员同值字段没有逐条对照价值"
            assert "Funtional advantage" not in b, "长文本字段聚合只会稀释语义"


def test_overview_quick_scan_line_groups_models_with_key_attrs():
    """概览要有「条目速览」块：每款带关键参数，分组式追问才有素材。"""
    res = parse_document("bridge.xlsx", _bridge_xlsx())
    assert res.blocks[0].startswith("【文档概览】")
    scan = res.blocks[1]
    assert scan.startswith("【条目速览】")
    assert "WB100: 5.8GHz" in scan
    assert "12.5KG" in scan


def test_attribute_summary_omitted_when_records_too_few():
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "参数"
    ws.append(["参数", "A1", "B2", "C3"])
    ws.append(["重量", "1KG", "2KG", "3KG"])
    buf = io.BytesIO()
    wb.save(buf)
    res = parse_document("小表.xlsx", buf.getvalue())
    assert not any(b.startswith("【属性对照】") for b in res.blocks)
    assert not any(b.startswith("【条目速览】") for b in res.blocks)


def test_unsupported_ext_lists_supported_formats():
    with pytest.raises(ParseError) as ei:
        parse_document("archive.zip", b"PK\x03\x04")
    msg = str(ei.value)
    assert "不支持的格式" in msg
    assert ".pdf" in msg, "报错里要列出支持哪些格式，否则用户只能猜"


def test_file_without_extension_is_rejected():
    with pytest.raises(ParseError):
        parse_document("noext", b"hello")


def test_empty_file_is_rejected():
    with pytest.raises(ParseError):
        parse_document("empty.txt", b"")


# --------------------------------------------------------------------------- #
# 纯文本类
# --------------------------------------------------------------------------- #
def test_txt_and_md_read_as_plain_text():
    assert parse_document("a.txt", "退货说明".encode("utf-8")).text == "退货说明"
    assert parse_document("a.md", "# 标题\n正文".encode("utf-8")).text.startswith("# 标题")


def test_gbk_encoded_txt_is_decoded():
    """国内导出的 txt 常是 GBK，不能变成乱码。"""
    text = parse_document("gbk.txt", "退货政策说明".encode("gb18030")).text
    assert text == "退货政策说明"


def test_crlf_and_extra_blank_lines_are_normalized():
    text = parse_document("n.txt", b"a\r\n\r\n\r\n\r\nb").text
    assert "\r" not in text
    assert "\n\n\n" not in text


def test_csv_becomes_header_prefixed_sentences():
    """CSV 转成「字段: 值」的自然语言行，否则向量检索对不上问题。"""
    text = parse_document("faq.csv", "问题,答案\n退货,7天无理由\n".encode("utf-8")).text
    assert "问题: 退货" in text
    assert "答案: 7天无理由" in text


def test_csv_without_data_rows_is_rejected():
    with pytest.raises(ParseError):
        parse_document("only_header.csv", "a,b\n".encode("utf-8"))


# --------------------------------------------------------------------------- #
# 结构化文档
# --------------------------------------------------------------------------- #
def _docx_bytes() -> bytes:
    import docx

    d = docx.Document()
    d.add_paragraph("退换货政策")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "项目"
    t.cell(0, 1).text = "时长"
    t.cell(1, 0).text = "整机保修"
    t.cell(1, 1).text = "24个月"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_reads_paragraphs_and_tables():
    """表格里的保修时长往往才是答案，不能只读段落丢掉表格。"""
    text = parse_document("policy.docx", _docx_bytes()).text
    assert "退换货政策" in text
    assert "整机保修" in text and "24个月" in text


def _xlsx_bytes() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "保修表"
    ws.append(["品类", "保修期"])
    ws.append(["户外电源", "36个月"])
    ws.append(["电池组", "12个月"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_reads_rows_with_sheet_name():
    text = parse_document("warranty.xlsx", _xlsx_bytes()).text
    assert "保修表" in text
    assert "户外电源" in text and "36个月" in text


def test_xlsx_emits_atomic_blocks_per_record():
    """表格解析必须给出原子块：一条记录一块，不许再被并段粘回去。"""
    res = parse_document("warranty.xlsx", _xlsx_bytes())
    assert len(res.blocks) == 2
    assert all("保修表" in b for b in res.blocks)
    assert "户外电源" in res.blocks[0] and "36个月" in res.blocks[0]
    # text 只作预览/兜底用，块之间以空行分隔
    assert "户外电源" in res.text and "电池组" in res.text


def test_transposed_sheet_yields_one_block_per_model():
    """转置参数表（列=型号）：型号与其全部参数必须落在同一块内。"""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Wireless bridge"
    ws.append(["参数", "WB730", "WB620E"])
    ws.append(["CTN weight", "15.5KG", "13KG"])
    ws.append(["PRICE(RMB)", "￥165", "￥145"])
    buf = io.BytesIO()
    wb.save(buf)

    res = parse_document("bridge.xlsx", buf.getvalue())
    assert len(res.blocks) == 2
    joined = res.blocks[0]
    assert "WB730" in joined and "15.5KG" in joined and "￥165" in joined


def test_text_formats_produce_path_prefixed_blocks():
    """文本类也产出结构化块，且每块带章节路径前缀（不再退化成纯滑窗）。"""
    res = parse_document("a.md", "# 标题\n\n正文一段。\n\n## 子节\n\n子节内容。\n".encode("utf-8"))
    assert res.blocks, "md 应产出结构化块"
    assert any(b.startswith("【标题】") for b in res.blocks)
    assert any(b.startswith("【标题 > 子节】") for b in res.blocks)


def test_docx_without_text_is_rejected():
    import docx

    buf = io.BytesIO()
    docx.Document().save(buf)
    with pytest.raises(ParseError):
        parse_document("blank.docx", buf.getvalue())


# --------------------------------------------------------------------------- #
# 表标题行：数据区之前的合并单元格大标题必须保留（品牌 grounding）
# --------------------------------------------------------------------------- #
def test_table_title_row_preserved():
    """首行独值大标题（如「AI Smart Wireless Bridge/CPE」）要进块，不能被当噪声丢掉。"""
    data = _make_xls(
        [["AI Smart Wireless Bridge/CPE"], [None]]
        + [["型号", "价格"], ["A", "1元"], ["B", "2元"], ["C", "3元"], ["D", "4元"], ["E", "5元"]]
    )
    result = parse_document("品牌表.xls", data)
    title_blocks = [b for b in result.blocks if "AI Smart" in b]
    assert title_blocks, "表标题行必须生成独立块"
    assert any("表标题" in b for b in title_blocks)


def test_single_column_list_not_mistaken_as_title():
    """单列长列表（每行只有 1 格）不能被整体误判成标题（维持原有的无数据行报错）。"""
    data = _make_xls([["名称"], ["甲"], ["乙"], ["丙"], ["丁"]])
    with pytest.raises(ParseError):
        parse_document("名单.xls", data)


# --------------------------------------------------------------------------- #
# 只读模式下 <dimension> 写错的 xlsx
#
# 真实踩坑：1688 订单导出文件把 dimension 声明成 "A1"，openpyxl 只读模式信任它，
# 631 行订单数据只进了 1 行 —— 上传显示成功，但 AI 怎么问都答"没查到资料"。
# --------------------------------------------------------------------------- #
def _make_xlsx_with_bogus_dimension(rows: list[list]) -> bytes:
    """造一个 xlsx，并把 sheet1 的 dimension 改写成错误的 A1。"""
    openpyxl = pytest.importorskip("openpyxl")
    import io as _io
    import re as _re
    import zipfile as _zf

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "sheet1"
    for r in rows:
        ws.append(r)
    buf = _io.BytesIO()
    wb.save(buf)
    raw = buf.getvalue()

    # 把 workbook 重新打包，只改 sheet1.xml 里的 dimension
    src = _zf.ZipFile(_io.BytesIO(raw))
    out = _io.BytesIO()
    with _zf.ZipFile(out, "w", _zf.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data = _re.sub(rb'<dimension ref="[^"]*"/>', b'<dimension ref="A1"/>', data)
            dst.writestr(name, data)
    return out.getvalue()


def test_bogus_dimension_still_reads_all_rows():
    """dimension 写错成 A1 时，仍必须读到全部数据行（不能被静默截断）。"""
    headers = ["订单编号", "实付款(元)", "订单状态"]
    body = [
        ["5127693301594110026", "30.88", "交易成功"],
        ["5127693301594110027", "128.00", "交易成功"],
        ["5127693301594110028", "66.50", "交易成功"],
        ["5127693301594110029", "19.90", "交易成功"],
        ["5127693301594110030", "240.00", "交易成功"],
        ["5127693301594110031", "88.00", "交易成功"],
    ]
    data = _make_xlsx_with_bogus_dimension([headers, *body])

    result = parse_document("订单导出.xlsx", data)
    joined = "\n".join(result.blocks)

    # 全部订单号都要进来，缺一个就说明被 dimension 截断了
    for row in body:
        assert row[0] in joined, f"订单 {row[0]} 丢失，疑似 dimension 截断"

    # 属性对照块也要按列聚合出全部金额
    attr_blocks = [b for b in result.blocks if "实付款" in b]
    assert attr_blocks, "应按列生成金额属性块"
    assert "30.88" in attr_blocks[0] and "240.00" in attr_blocks[0]


# ---------- 纯文本参数区（【参数对照】块） ----------

from app.textparse import _build_kv_param_summaries  # noqa: E402

_P33_BLOCK = """5. 产品参数
• 核准代码体现在产品铭牌上。
• 因产品持续改善的需要，本资料产品以实物为准，我公司保留产品更新的权利。
在使用中如遇任何问题，请用微信扫描如下二维码，联系在线客服。
产品型号 DDX19 / DDX29
主机额定输入 20V 2A
充电时间 约 5h
自动洗拖布集尘座型号 CH2358/CH2358A
额定输入 220V ～ 50Hz
额定输出 20V 2A
额定输入电流（充电状
态） 0.5A
功率（集尘状态） 1000W
功率（热水洗拖布状态） 1500W
自动上下水模块型号 FM2243
额定输入电压 12V
额定输入电流 0.6A
最大进水水压 0.7MPa
主机尺寸 ( 宽深高 mm) 313mm×346mm×95mm
基站尺寸 ( 宽深高 mm) 394mm×443mm×527.5mm
在使用中如遇任何问题，请联系我们的售后服务中心： 400-886-8888，将有专业人员为您解答问题。
更多详情请至科沃斯官网： www.ecovacs.com.
"""

_PROSE_BLOCK = """2
注意
1. 针对产品在清扫中可能出现的问题，应及早排除。清理地面上的电源线
和细小物品避免产品在清洁过程中受阻。将地毯的边穗翻折到地毯
下，并使垂挂的窗帘、桌布等不要接触地面。
2. 若存在诸如楼梯等悬空环境，请先测试产品看其是否可以检测到悬空
区域边缘而不跌落。应在悬空区域边缘设置防护栏以防产品跌落。
"""


def test_kv_param_summary_extracts_manual_params():
    """说明书参数区的「键 值」行必须聚成一条参数对照块。"""
    out = _build_kv_param_summaries("说明书.pdf", [_P33_BLOCK])
    assert out, "成片参数区应生成参数对照块"
    block = "\n".join(out)
    assert "产品型号：DDX19 / DDX29" in block
    assert "充电时间：约 5h" in block
    assert "最大进水水压：0.7MPa" in block
    assert "313mm×346mm×95mm" in block


def test_kv_param_summary_merges_line_wrapped_key():
    """PDF 折行把键切断（'额定输入电流（充电状'+'态） 0.5A'）必须拼回完整键。"""
    out = _build_kv_param_summaries("说明书.pdf", [_P33_BLOCK])
    block = "\n".join(out)
    assert "额定输入电流（充电状态）：0.5A" in block


def test_kv_param_summary_skips_prose_and_isolated_pairs():
    """散文页不生成块；孤立的零星键值行（±5 行内不足 3 个命中）不收录。"""
    assert _build_kv_param_summaries("说明书.pdf", [_PROSE_BLOCK]) == []
    isolated = "产品型号 DDX19 / DDX29\n这是很长的正文一句话没有别的参数。\n又一句正文内容依然没有参数。"
    assert _build_kv_param_summaries("说明书.pdf", [isolated]) == []


def test_kv_param_summary_requires_min_pairs():
    """全文档不足 5 对参数不生成块（正常检索即可覆盖，避免噪声块）。"""
    small = "\n".join(
        [
            "产品型号 DDX19 / DDX29",
            "充电时间 约 5h",
            "额定输入电压 12V",
            "额定输入电流 0.6A",
            "正文一句话收尾。",
        ]
    )
    assert _build_kv_param_summaries("说明书.pdf", [small]) == []


# ---------- 散文要点区（【要点清单】块）与售后联系方式（【售后服务】块） ----------

from app.textparse import _build_bullet_summaries, _build_service_summary  # noqa: E402

_P14_BLOCK = """1. 连接电源线
*语音仅X5 PRO标配
主机面盖下方
5 获取完整版使用指南
微信扫描机身二维码，获取完整版使用指南。
• 全能基站附近若有镜子、 反光的踢脚线等反光的物体， 需遮挡其底部 14cm 的部分。
• 请勿将全能基站放在阳光直射的地方。
• 建议放置在 Wi-Fi 信号强的位置，以便获得更好的使用体验。
2. 寻找合适的位置放置全能基站
全能基站靠墙放置在平坦的硬质地面，左右0.1m和前方0.8m范围内不要
放置物品。
≥0.1m (0.33')
3. 使用水箱
"""


def test_bullet_summary_extracts_placement_requirements():
    """摆放要点（bullet 密集区）必须单独成块，并吸收紧邻的靠墙放置要求句。"""
    out = _build_bullet_summaries("说明书.pdf", [_P14_BLOCK])
    assert out, "成片 bullet 要求应生成要点清单块"
    block = "\n".join(out)
    assert block.startswith("【要点清单】")
    assert "14cm" in block
    assert "阳光直射" in block
    # 折行要求句必须拼回完整（"…范围内不要"+"放置物品。"）
    assert "左右0.1m和前方0.8m范围内不要放置物品" in block
    # 步骤标题不得混进句子（"2. 寻找合适的位置…"）
    assert "2. 寻找合适的位置" not in block
    # 间距标注行（无要求关键词）不得混入
    assert "≥0.1m (0.33')" not in block


def test_bullet_summary_skips_scattered_bullets():
    """孤立 bullet（±5 行内不足 3 条）不生成块。"""
    lonely = "正文一句话。\n• 孤立的一条要求请不要忽视它。\n正文一句话。\n正文一句话。"
    assert _build_bullet_summaries("说明书.pdf", [lonely]) == []


def test_service_summary_extracts_hotline_website_wechat():
    """售后热线/官网/微信在线客服必须聚成一条独立的【售后服务】块。"""
    out = _build_service_summary("说明书.pdf", [_P33_BLOCK])
    assert len(out) == 1
    block = out[0]
    assert block.startswith("【售后服务】")
    assert "400-886-8888" in block
    assert "www.ecovacs.com" in block
    assert "在线客服" in block


def test_service_summary_omitted_without_contact_info():
    """无任何联系方式时不生成块。"""
    plain = "产品型号 DDX19\n这是很长的正文一句话没有联系方式。\n又一句正文内容。"
    assert _build_service_summary("说明书.pdf", [plain]) == []


# --------------------------------------------------------------------------- #
# 扫描版 PDF：整页视觉转写兜底
# --------------------------------------------------------------------------- #
def _two_page_pdf(first_page_lines: list[str]) -> bytes:
    """第 1 页有文字层、第 2 页为空的混合 PDF。"""
    ops = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
    for i, ln in enumerate(first_page_lines):
        esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i:
            ops.append("T*")
        ops.append(f"({esc}) Tj")
    ops.append("ET")
    s1 = "\n".join(ops).encode("latin-1")
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(s1)).encode() + b">\nstream\n" + s1 + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R >>",
            b"<< /Length 0 >>\nstream\nendstream",
        ]
    )


def test_scanned_pdf_ocr_fallback_transcribes(monkeypatch):
    """无文字层 + 视觉模型可用：整页转写结果入库，页码记录进 ocr_pages。"""
    import app.textparse as tp

    monkeypatch.setattr("app.vision.enabled", lambda: True)
    monkeypatch.setattr(
        tp,
        "_ocr_pdf_pages",
        lambda data, pages: ({1: "额定热负荷 20.0kW，适用水压 0.02~1.0MPa"}, []),
    )
    result = parse_document("scan.pdf", _empty_pdf())
    assert "额定热负荷 20.0kW" in result.text
    assert result.ocr_pages == {1}
    assert any("转写" in w for w in result.warnings)


def test_mixed_pdf_ocr_only_empty_pages(monkeypatch):
    """混合 PDF：有文字层的页原样保留，只对无文字页做转写，且按页码归位。"""
    import app.textparse as tp

    called: list[list[int]] = []

    def fake_ocr(data, pages):
        called.append(list(pages))
        return {2: "第二页扫描转写内容：保修六年"}, []

    monkeypatch.setattr("app.vision.enabled", lambda: True)
    monkeypatch.setattr(tp, "_ocr_pdf_pages", fake_ocr)
    result = parse_document("mixed.pdf", _two_page_pdf(["Warranty 36 months"]))
    assert "Warranty 36 months" in result.text
    assert "保修六年" in result.text
    assert result.ocr_pages == {2}
    assert called and called[0] == [2], "有文字层的第 1 页不该送 OCR"


def test_scanned_pdf_ocr_fails_gives_reason(monkeypatch):
    """视觉模型可用但识别颗粒无收：报错必须带原因和下一步建议。"""
    import app.textparse as tp

    monkeypatch.setattr("app.vision.enabled", lambda: True)
    monkeypatch.setattr(tp, "_ocr_pdf_pages", lambda data, pages: ({}, ["视觉模型调用失败"]))
    with pytest.raises(ParseError) as ei:
        parse_document("scan.pdf", _empty_pdf())
    msg = str(ei.value)
    assert "文字层" in msg
    assert "视觉模型调用失败" in msg
    assert "DOCX" in msg or "TXT" in msg


def test_scanned_pdf_without_vision_mentions_provider():
    """视觉未启用时的报错要点名 VISION_PROVIDER，让管理员知道去哪里开。"""
    with pytest.raises(ParseError) as ei:
        parse_document("scan.pdf", _empty_pdf())
    msg = str(ei.value)
    assert "VISION_PROVIDER" in msg
    assert "DOCX" in msg or "TXT" in msg
