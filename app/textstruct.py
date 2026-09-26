"""文本结构化：把 md / txt / docx / pdf 切成**带层级路径**的语义块。

为什么单独一层：按字数滑窗切出来的块没有主语、没有归属，模型拿到也读不懂。
检索单元的自足性主要靠两件事——「这块属于哪个章节」+「这块讲的是哪个条目」。
所以这里产出的是 (路径, 标题, 正文) 三元组，渲染时把路径写成前缀：

    【产品中心 > 无线网桥 > WB730】
    参数
    Chipset: MTK7621；CTN weight: 15.5KG

切分原则（与表格层一致）：
- **先按结构切，再谈字数**：标题、代码块、表格是天然边界，绝不从中间切断；
- 字数只在「结构块本身过大」时才介入（交给 rag 的滑窗，且滑窗会补回路径前缀）；
- 过短的相邻块允许合并，但**只在同父路径下合并**，跨章节不合并。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 模式
# --------------------------------------------------------------------------- #
_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_EQ = re.compile(r"^=+\s*$")
_SETEXT_DASH = re.compile(r"^-{2,}\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_DELIM = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
_BULLET = re.compile(r"^\s*(?:[-*+•·]|[0-9]+[.)、]|[（(][0-9]+[)）])\s+")

# Markdown 图片 `![alt](url)`。alt 常被写成「图 2-3 设备安装示意图」，是有用的描述。
_IMAGE = re.compile(r"^!\[([^\]]*)\]\([^)]*\)")
# 题注：图/表 + 编号 + 说明。docx 里图片下一段常是它，是白捡的图描述。
_CAPTION = re.compile(
    r"^\s*(?:图|图表|表|Figure|Fig|Table|IMAGE|图片)\s*"
    r"[0-9]+(?:[.\-–—][0-9]+)*\s*[:：、.\s]*.{0,60}$"
)

# 无格式文本（txt / pdf）的标题线索。刻意保守：宁可漏判，不可把正文当标题。
# 单级数字编号（"1. xxx"）不算标题——列表项太容易误判；
# 至少要两级（"1.2 xxx"）或中文序号（"第三章"、"一、"）。
_TXT_HEAD = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9一二三四五六七八九十百千]+\s*[章节条篇部分]"
    r"|[0-9]+(?:\.[0-9]+){1,3}(?=[\s、．.]|$)"
    r"|[一二三四五六七八九十]+\s*[、．.]"
    r")"
)

_SENT_END = "。！？!?；;\n"
# 行尾是这些字符时说明句子已完整，不应与下一行拼接（PDF 断行还原用）
_LINE_CONTINUE_STOP = tuple("。！？!?；;：:”」』）)")


@dataclass
class StructBlock:
    path: list[str] = field(default_factory=list)  # 祖先标题（不含自身）
    title: str = ""                                 # 自身标题（可能为空）
    body: str = ""
    atomic: bool = False                            # True=独立成块，不参与合并


def _clean(line: str) -> str:
    return line.replace("\u3000", " ").rstrip()


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def render_blocks(
    blocks: list[StructBlock],
    *,
    max_chars: int = 600,
    min_merge: int = 120,
    max_depth: int = 4,
) -> list[str]:
    """把结构化块渲染成入库文本。

    合并规则（宁可不并，不可并错）：
    - 无标题的段落之间可以并（等同旧 chunk_text 的段落并段）；
    - **顶层标题块不并**——两章的父路径都为空，若按空父路径比较会把整篇并成一块；
    - 二级及以下按**父路径**并，跨父路径绝不并；
    - 缓冲区装到 `min_merge` 就收手，避免把整个文档并成一块。
    合并块的各段标题会写进正文，所以并出来的块仍分得清谁是谁。
    """
    out: list[str] = []
    parent: list[str] | None = None
    key: tuple | None = None
    items: list[tuple[str, str]] = []
    size = 0

    def _prefix(p: list[str]) -> str:
        if not p:
            return ""
        return "【" + " > ".join(p[-max_depth:]) + "】\n"

    def flush() -> None:
        nonlocal parent, key, items, size
        if not items:
            return
        if len(items) == 1:
            # 单块：标题并进前缀，正文不重复写
            title, body = items[0]
            head = _prefix(list(parent or []) + ([title] if title else []))
            text = (head + body).strip()
        else:
            # 合并块：前缀只到父级，各段标题写进正文，才分得清谁是谁
            head = _prefix(parent or [])
            body = "\n\n".join(f"{t}\n{b}" if t else b for t, b in items)
            text = (head + body).strip()
        if text:
            out.append(text)
        parent, key, items, size = None, None, [], 0

    for blk in blocks:
        body = (blk.body or "").strip()
        if not body and not blk.title:
            continue
        if blk.atomic:
            flush()
            head = _prefix(blk.path)
            text = (head + body).strip()
            if text:
                out.append(text)
            continue

        # 前缀写**完整路径**（含自身标题）：块被滑窗切开时 rag 只补这一个前缀，
        # 若前缀不含自身标题，第二片就只剩父路径、丢了小节名。
        parent_path = list(blk.path[:-1])
        item = ((blk.path[-1] if blk.path else blk.title).strip(), body)
        cur = len(item[0]) + len(item[1])

        if not blk.path:
            cur_key: tuple | None = ("P",)          # 无标题段落：彼此可并
        elif len(blk.path) == 1:
            cur_key = None                          # 顶层标题：独占一块
        else:
            cur_key = ("H", tuple(blk.path[:-1]))   # 二级及以下：按父路径并

        if (
            items
            and cur_key is not None
            and key == cur_key
            and parent == parent_path
            and size < min_merge
            and size + cur <= max_chars
        ):
            items.append(item)
            size += cur
        else:
            flush()
            parent = parent_path
            key = cur_key
            items = [item]
            size = cur
    flush()
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def parse_markdown(text: str, max_chars: int = 600) -> list[StructBlock]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[StructBlock] = []
    stack: list[tuple[int, str]] = []   # (level, title)，即当前祖先链
    buf: list[str] = []

    def path() -> list[str]:
        return [t for _, t in stack]

    def flush(atomic: bool = False, title: str = "") -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body and not title:
            return
        blocks.append(StructBlock(path=path(), title=title, body=body, atomic=atomic))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 代码块：整块保留，绝不从中间切
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker = fence.group(1)[0]
            buf = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(marker * 3):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            flush(atomic=True)
            continue

        # 图片：`![alt](url)`。alt 是**免费的图描述**，务必留下来 ——
        # 图本身进不了向量库（embedding 是纯文本的），但「图 2-3 设备安装示意图」
        # 这种 alt 能让模型在被问到时指路，而不是顺着上下文编。
        m_img = _IMAGE.match(stripped)
        if m_img:
            flush()
            blocks.append(
                StructBlock(path=path(), body=image_placeholder(m_img.group(1)), atomic=True)
            )
            i += 1
            continue

        # 表格：与 xlsx 走同一套结构化渲染（一行一块，带列名），不自创格式
        if stripped.startswith("|") and i + 1 < n and _TABLE_DELIM.match(lines[i + 1] or ""):
            flush()
            header = stripped
            rows: list[str] = []
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            for piece in _table_blocks(header, rows, max_chars):
                blocks.append(StructBlock(path=path(), body=piece, atomic=True))
            continue

        # ATX 标题
        m = _ATX.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buf = []
            i += 1
            continue

        # Setext 标题（下一行是 === 或 ---）。
        # 必须**紧邻**上一行才成立：Markdown 里 `---` 更常见的用法是水平分割线，
        # 它前后必有空行。不加这个判据，分割线会把上一行吃掉当标题、内容凭空少一段。
        if stripped and i + 1 < n and buf and buf[-1] == _clean(lines[i - 1]):
            nxt = lines[i + 1].strip()
            if _SETEXT_EQ.match(nxt):
                flush()
                level = 1
                stack.clear()
                stack.append((level, stripped))
                buf = []
                i += 2
                continue
            if _SETEXT_DASH.match(nxt) and not _BULLET.match(stripped):
                flush()
                while stack and stack[-1][0] >= 2:
                    stack.pop()
                stack.append((2, stripped))
                buf = []
                i += 2
                continue

        if not stripped:
            # 空行只做段落分隔，不结束小节
            if buf and buf[-1] != "":
                buf.append("")
            i += 1
            continue

        buf.append(_clean(line))
        i += 1

    flush()
    return blocks


def image_placeholder(desc: str = "") -> str:
    """图片的占位文本。

    图本身进不了向量库（embedding 是纯文本的），但必须**留下"这里有张图、画的是什么"**：
    否则用户问到图里的内容时，模型手里没有任何线索，只能顺着上下文编。
    有描述就带上（md 的 alt、docx 的题注都是白捡的），没有就如实说无法提取。
    """
    desc = (desc or "").strip()
    if desc:
        return f"【图】{desc}"
    return "【图】文档此处有一张图片，图中内容无法提取"


_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _has_image(element) -> bool:
    """段落里是否嵌了**带关系引用**的图片（图片不产生 run text，所以 text 为空）。

    只看「有 pict/blip 字样」会把大量假图放进来：Word 的水平分隔线是
    `<v:rect o:hr="t"><v:imagedata o:title=""/></v:rect>`，文本框、形状也在
    pict/graphicData 里 —— 它们都没有 r:embed / r:id 引用，不是图。
    （实测一份 PRD 文档里 28 个"图"全是分隔线，一张真图都没有。）
    所以判定收紧到：blip / imagedata 元素上**真的挂着关系引用**才算图。
    """
    try:
        for el in element.iter():
            if not isinstance(el.tag, str):
                continue
            tag = el.tag.rsplit("}", 1)[-1]
            if tag not in ("blip", "imagedata"):
                continue
            if any(attr.startswith(_REL_NS) for attr in el.attrib):
                return True
    except Exception:  # pragma: no cover - 结构异常按无图处理
        return False
    return False


def _pack_rows(pieces: list[str], max_chars: int, short: int = 120) -> list[str]:
    """短记录打包成组，长记录保持独立。

    说明性小表（如「层级 / 能力 / 对应章节」）一行一块会产生几十个几十字的碎块，
    既挤占 top_k 名额又抬高向量化成本。但**长记录不能并**——一条几百字的产品参数
    记录本身就是独立主题，并在一起会稀释相似度（FAQ 并块 0.78→0.51 的老问题）。
    所以只有短记录参与打包。
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("\n".join(buf))
            buf.clear()

    for p in pieces:
        if len(p) > short:
            flush()
            out.append(p)
            continue
        if buf and sum(len(x) for x in buf) + len(p) + len(buf) > max_chars:
            flush()
        buf.append(p)
    flush()
    return out


def _table_blocks(header: str, rows: list[str], max_chars: int) -> list[str]:
    """Markdown 表格 → 与 xlsx 相同的记录块（一行一块，带列名）。

    整表塞一块会稀释相似度（"6 条 FAQ 并一块 0.78→0.51" 的老问题），
    所以这里拆成行级记录；只有判定不出形态时才退回整表。
    """
    if not rows:
        return [header] if header.strip() else []

    def cells(row: str) -> list[str]:
        body = row.strip().strip("|")
        return [c.strip() for c in body.split("|")]

    grid = [cells(header)] + [cells(r) for r in rows]
    try:
        from app.tablelayout import parse_sheet

        pieces, _ = parse_sheet(grid, max_chars=max_chars)
        if pieces:
            return _pack_rows(pieces, max_chars)
    except Exception:  # pragma: no cover - 判定失败不该让整篇解析挂掉
        pass

    # 退回整表，按行分组重复表头
    out: list[str] = []
    cur = header
    for row in rows:
        if len(cur) + len(row) + 1 > max_chars and cur != header:
            out.append(cur)
            cur = header
        cur = f"{cur}\n{row}"
    if cur.strip():
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# 无格式文本（txt / 清洗后的 pdf）
# --------------------------------------------------------------------------- #
def parse_plain_text(text: str) -> list[StructBlock]:
    """靠空行分段 + 保守的编号标题识别。猜不出结构时退化为段落块（仍带路径）。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[StructBlock] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if body:
            blocks.append(StructBlock(path=[t for _, t in stack], body=body))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf and buf[-1] != "":
                buf.append("")
            continue
        if _is_txt_heading(stripped):
            flush()
            # 章 > 节 用编号里的点数判断（1.2 → 2 级，1.2.3 → 3 级）
            depth = _head_depth(stripped)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, stripped))
            continue
        buf.append(_clean(line))
    flush()
    return blocks


def _is_txt_heading(line: str) -> bool:
    if len(line) > 40:
        return False
    if line[-1] in _LINE_CONTINUE_STOP:
        return False
    if not _TXT_HEAD.match(line):
        return False
    # 标题后面得有内容（"1.2 " 这种光秃秃的不是标题）
    rest = _TXT_HEAD.sub("", line).strip()
    return bool(rest) or len(line) <= 12


def _head_depth(line: str) -> int:
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*)", line)
    if m:
        return min(m.group(1).count(".") + 1, 4)
    if line.startswith("第") and ("章" in line[:6] or "篇" in line[:6]):
        return 1
    if line.startswith("第") and ("节" in line[:6] or "条" in line[:6]):
        return 2
    return 3


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def parse_docx(document, max_chars: int = 600) -> tuple[list[StructBlock], list[str]]:
    """按**文档顺序**遍历段落与表格（旧实现先段落后表格，顺序是错的）。

    标题层级取自 style.name（Heading N / 标题 N / Title），取不到时退回 outlineLvl。
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    warnings: list[str] = []
    blocks: list[StructBlock] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if body:
            blocks.append(StructBlock(path=[t for _, t in stack], body=body))

    try:
        children = list(document.element.body.iterchildren())
    except Exception as exc:  # pragma: no cover - 结构异常不该让上传 500
        warnings.append(f"文档结构读取异常，已按纯段落处理：{exc}")
        children = []

    image_count = 0
    pending_image: StructBlock | None = None   # 上一张图，等题注来认领

    for child in children:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = (para.text or "").strip()
            if not text:
                # 图片不产生 run text，所以空段落很可能是图片 —— 之前直接 continue，
                # 等于静默丢图：用户上传产品手册，参数表截图全没了却毫无提示。
                if _has_image(child):
                    flush()
                    pending_image = StructBlock(
                        path=[t for _, t in stack], body=image_placeholder(""), atomic=True
                    )
                    blocks.append(pending_image)
                    image_count += 1
                continue
            # 题注（"图 2-3 设备安装示意图"）紧跟图片时，拿来当图描述，
            # 比只留一个光秃秃的占位符有用得多
            if pending_image is not None and _CAPTION.match(text) and len(text) <= 80:
                pending_image.body = image_placeholder(text)
                pending_image = None
                continue
            pending_image = None
            level = _docx_heading_level(para)
            if level:
                flush()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                continue
            buf.append(text)
        elif tag == "tbl":
            flush()
            try:
                table = Table(child, document)
                grid = [[c.text.strip() for c in row.cells] for row in table.rows]
            except Exception:  # pragma: no cover
                continue
            if not grid or not any(any(v for v in r) for r in grid):
                continue
            from app.tablelayout import parse_sheet

            pieces, tw = parse_sheet(grid, max_chars=max_chars)
            for w in tw:
                warnings.append(f"表格：{w}")
            for piece in pieces or [_rows_fallback(grid)]:
                if piece and piece.strip():
                    blocks.append(
                        StructBlock(path=[t for _, t in stack], body=piece.strip(), atomic=True)
                    )
    flush()
    if image_count:
        warnings.append(
            f"文档含 {image_count} 张图片，图中内容无法提取（已在原位留下占位说明）。"
            f"如果答案藏在图里，AI 会如实说读不到图中内容，而不是猜。"
        )
    return blocks, warnings


def _docx_heading_level(para) -> int:
    name = ""
    try:
        name = (para.style.name or "") if para.style is not None else ""
    except Exception:  # pragma: no cover
        name = ""
    low = name.lower()
    m = re.search(r"heading\s*(\d)", low)
    if m:
        return min(int(m.group(1)), 6)
    m = re.search(r"标题\s*(\d)", name)
    if m:
        return min(int(m.group(1)), 6)
    if "title" in low or name == "标题":
        return 1
    if "subtitle" in low or "副标题" in name:
        return 2
    # 退回 outlineLvl（w:pPr/w:outlineLvl/@w:val，0-based）
    try:
        ppr = para._p.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr"
        )
        if ppr is not None:
            node = ppr.find(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl"
            )
            if node is not None:
                val = node.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                )
                if val is not None:
                    return min(int(val) + 1, 6)
    except Exception:  # pragma: no cover
        pass
    return 0


def _rows_fallback(grid: list[list[str]]) -> str:
    return "\n".join("；".join(v for v in row if v) for row in grid if any(row))


# --------------------------------------------------------------------------- #
# PDF：剥页眉页脚 + 断行还原
# --------------------------------------------------------------------------- #
def clean_pdf_pages(pages: list[str]) -> tuple[str, list[str]]:
    """返回 (清洗后的全文, warning 列表)。

    两件事：
    1. 剥页眉页脚——跨页重复出现的短行（前 2 行 / 后 2 行）判为页眉页脚并删除；
    2. 断行还原——PDF 提取是按视觉行断的，行尾不在句末时把下一行拼回来，
       否则每个 chunk 里都是半截句子（这是 pdf 文档召回差的头号原因）。
    """
    warnings: list[str] = []
    page_lines = [
        [ln.strip() for ln in (pg or "").split("\n") if ln.strip()] for pg in pages
    ]
    page_lines = [p for p in page_lines if p]
    if not page_lines:
        return "", warnings

    n_pages = len(page_lines)

    def norm(s: str) -> str:
        # 数字统一替换：页眉常带「第 1 页 / 第 2 页」，字面上每页都不同，
        # 不去数字就永远统计不到重复，页眉也就剥不掉。
        return re.sub(r"\d+", "#", re.sub(r"\s+", "", s))

    # 1) 统计候选项（每页首 2 行 / 末 2 行）
    counter: dict[str, int] = {}
    for lines in page_lines:
        cand = lines[:2] + (lines[-2:] if len(lines) > 4 else [])
        for s in {norm(c) for c in cand}:
            if not s or len(s) > 60:
                continue
            counter[s] = counter.get(s, 0) + 1
    need = 2 if n_pages <= 2 else max(3, (n_pages * 3 + 4) // 5)
    boiler = {s for s, c in counter.items() if c >= need}
    if boiler:
        warnings.append(f"已剥离 {len(boiler)} 个跨页重复的页眉/页脚行")

    # 2) 逐页删除 + 断行还原
    cleaned: list[str] = []
    for lines in page_lines:
        kept: list[str] = []
        for idx, ln in enumerate(lines):
            if norm(ln) in boiler and (idx < 2 or idx >= len(lines) - 2):
                continue
            kept.append(ln)
        cleaned.append(_join_wrapped_lines(kept))
    return "\n\n".join(p for p in cleaned if p), warnings


def _join_wrapped_lines(lines: list[str]) -> str:
    """把被 PDF 切断的行拼回完整句子。

    判据用**页面宽度**而不是「行短不像句子」：PDF 是按排版宽度断行的，
    一行被写满（长度接近该页最长行）说明它是被切断的，而不是自然结束。
    靠"短行像标题"来判会误伤——被切断的中文片段恰好也无标点、也短。
    """
    if not lines:
        return ""
    widest = max(len(x) for x in lines)
    threshold = int(widest * 0.85)
    out: list[str] = []
    for ln in lines:
        if not out:
            out.append(ln)
            continue
        prev = out[-1]
        if _should_join(prev, ln, threshold):
            sep = "" if _is_cjk(prev[-1]) or _is_cjk(ln[0]) else " "
            out[-1] = f"{prev}{sep}{ln}"
        else:
            out.append(ln)
    return "\n".join(out)


def _should_join(prev: str, nxt: str, threshold: int) -> bool:
    if not prev or not nxt:
        return False
    if prev[-1] in _LINE_CONTINUE_STOP:
        return False
    # 下一行是列表项或标题 → 不拼（标题必须独立成行）
    if _BULLET.match(nxt) or _TXT_HEAD.match(nxt):
        return False
    return len(prev) >= threshold


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
