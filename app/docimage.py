"""从文档中抽出图片字节 —— 「图转文」链路的第一步。

为什么必须先把图抽出来
----------------------
embedding 是纯文本的，图片本身进不了向量库。文档里的图若不处理，入库时只剩
一句「此处有一张图片，图中内容无法提取」，客户问到图里的参数、表格、文字时，
模型手里没有任何线索 —— 它要么答不上来，要么顺着上下文编。

把图交给视觉模型读成文字，那些字才真正进库，检索才有东西可命中。

这**不是以图搜图**
------------------
「以图搜图」（多模态 embedding）解决的是「客户发一张图，找库里相似商品的图」，
要在检索端把图和图放进同一个向量空间，需要换 embedding、可能还要建独立的图片索引。

这里做的完全不同：**只在入库阶段用一次视觉能力**，把图一次性翻译成文本。
检索、重排、回答仍然是纯文本链路，向量库、embedding、rerank 全都不用动。

抽取结果怎么和正文对位
----------------------
不需要在文本里插锚点。约定：

    占位块在 blocks 中的出现顺序 == ImageRef.seq 的顺序

docx / md 都能保证这个顺序（正文遍历顺序就是图片出现顺序），
所以上层按「第 K 个占位块 ↔ 第 K 张图」配对即可。

数量一旦对不上（比如表格单元格里还藏了图），配对就整体错位，
宁可用**错 pairing**把 A 图的描述安到 B 图上更糟 —— 所以对不齐时上层必须放弃替换，
而不是«取前 N 个»硬配。
"""
from __future__ import annotations

import hashlib
import io
import re
import struct
from dataclasses import dataclass, field
from typing import Iterable

# 图片太小、字节太少，几乎只能是装饰元素（分隔线、图标、页眉 logo），
# 识别它们是纯浪费 token，还会白白占掉一个 chunk 名额、稀释检索结果。
MIN_BYTES = 2048
MIN_PIXELS = 64

# 下载 md 外链图片的上限：一张图拖垮整篇文档入库是不划算的
MAX_REMOTE_BYTES = 8 * 1024 * 1024
REMOTE_TIMEOUT = 8.0


@dataclass
class ImageRef:
    """一张被抽出的图。seq 从 1 开始，与占位块顺序对应。"""

    seq: int
    data: bytes
    mime: str
    ext: str
    source: str            # docx / pdf / md
    alt: str = ""          # md 的 alt 文本，白捡的上下文
    page: int | None = None
    note: str = ""

    @property
    def sha1(self) -> str:
        return hashlib.sha1(self.data).hexdigest()


@dataclass
class ExtractResult:
    images: list[ImageRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # 文档里出现了图、但一张都没抽出（全被过滤或解析失败）时为 True，
    # 上层据此给出「图没有被读懂」的确定性提示，而不是悄悄丢掉。
    saw_image_marker: bool = False


# --------------------------------------------------------------------------- #
# 图片元信息
# --------------------------------------------------------------------------- #
def image_size(data: bytes) -> tuple[int, int]:
    """读 PNG/JPEG/GIF/WEBP/BMP 的宽高。读不出返回 (0, 0)，不抛异常。

    刻意**不依赖 Pillow**：抽图是在上传主链路上跑的，多一个可选依赖就多一种
    「别人机器上跑不起来」的失败方式。这几种格式的头解析都是几行的事。
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        if data[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return int(w), int(h)
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                i += 2 + seg
            return 0, 0
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            w, h = struct.unpack("<HH", data[6:10])
            return int(w), int(h)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            # VP8 / VP8L / VP8X，够用即可
            if data[12:16] == b"VP8X" and len(data) >= 30:
                w = int.from_bytes(data[24:27], "little") + 1
                h = int.from_bytes(data[27:30], "little") + 1
                return w, h
            if data[12:16] == b"VP8L" and len(data) >= 25:
                bits = int.from_bytes(data[21:25], "little")
                return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            if data[12:16] == b"VP8 " and len(data) >= 30:
                w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
                h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
                return w, h
        if data[:2] == b"BM" and len(data) >= 26:
            w, h = struct.unpack("<ii", data[18:26])
            return abs(int(w)), abs(int(h))
    except Exception:  # noqa: BLE001 - 头解析失败一律当未知尺寸，不阻断抽图
        pass
    return 0, 0


_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


def mime_for(ext: str) -> str:
    return _MIME_BY_EXT.get((ext or "").lower().lstrip(".").replace("jpg", "jpeg"), "image/png")


def _decorative(data: bytes) -> bool:
    """是否是可以直接丢掉的装饰性图形。"""
    if len(data) < MIN_BYTES:
        return True
    w, h = image_size(data)
    if w and h and min(w, h) < MIN_PIXELS:
        return True
    return False


# --------------------------------------------------------------------------- #
# 各格式抽取
# --------------------------------------------------------------------------- #
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _docx_image_rids(document) -> list[str]:
    """按正文顺序收集图片关系 ID。用 lxml 完全限定属性名，不依赖序列化前缀。

    为什么不用正则匹配 `r:embed="..."`：lxml 序列化时的前缀不受我们控制，
    且分隔线/形状的空 <v:imagedata> 根本没有关系引用 —— 用属性名匹配
    既躲开前缀问题，又天然过滤掉这些假图。
    """
    rids: list[str] = []
    for el in document.element.body.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("blip", "imagedata"):
            continue
        rid = (
            el.attrib.get(f"{_REL_NS}embed")
            or el.attrib.get(f"{_REL_NS}id")
            or el.attrib.get(f"{_REL_NS}link")
        )
        if rid and rid not in rids:
            rids.append(rid)
    return rids


def _from_docx(data: bytes) -> tuple[list[tuple[bytes, str, str]], list[str]]:
    """docx：按正文顺序抽出每张图，返回 [(字节, 扩展名, rId)]。

    顺序必须来自**正文**，而不是 related_parts 的字典顺序：dict 是无序集合，
    拿它的顺序当出现顺序，图片描述就会整段错位。
    """
    try:
        import docx
    except ImportError:  # pragma: no cover
        return [], ["服务端未安装 DOCX 解析组件（python-docx），未能抽出文档中的图片"]

    try:
        document = docx.Document(io.BytesIO(data))
        rids = _docx_image_rids(document)
        parts = document.part.related_parts
    except Exception as exc:  # noqa: BLE001
        return [], [f"文档中的图片未能抽出：{exc}"]

    out: list[tuple[bytes, str, str]] = []
    for rid in rids:
        part = parts.get(rid)
        blob = getattr(part, "blob", None)
        if not blob:
            continue
        name = str(getattr(part, "partname", "") or "")
        if "/" in name:
            name = name.rsplit("/", 1)[-1]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in _MIME_BY_EXT:
            ext = "jpeg" if blob[:2] == b"\xff\xd8" else "png"
        out.append((blob, ext, rid))
    return out, []


def _from_pdf(data: bytes) -> tuple[list[tuple[bytes, str, int]], list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        return [], ["服务端未安装 PDF 解析组件（pypdf），未能抽出文档中的图片"]

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return [], [f"文档中的图片未能抽出：{exc}"]

    out: list[tuple[bytes, str, int]] = []
    try:
        for idx, page in enumerate(reader.pages, start=1):
            for image in page.images:
                blob = getattr(image, "data", None)
                if not blob:
                    continue
                fmt = str(getattr(image, "image_format", "") or "").lower()
                ext = fmt if fmt in _MIME_BY_EXT else ""
                if not ext:
                    ext = "jpeg" if blob[:2] == b"\xff\xd8" else "png"
                out.append((blob, ext, idx))
    except Exception as exc:  # noqa: BLE001 - 个别页抽图失败不该连累整篇文档
        return out, [f"第 {len(out) + 1} 页之后的图片未能抽出：{exc}"]
    return out, []


_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")


# --------------------------------------------------------------------------- #
# 扫描版 PDF：把**整页**渲染成图片（页面级 OCR 兜底的第一步）
# --------------------------------------------------------------------------- #
def render_pdf_pages(
    data: bytes, page_nos: list[int], *, dpi: int = 150
) -> tuple[list[tuple[int, bytes, str]], list[str]]:
    """把指定页渲染成 JPEG，供视觉模型做整页文字转写。

    与 _from_pdf 的区别：_from_pdf 抽的是**内嵌图片对象**（插图、页眉 logo），
    这里渲染的是**页面本身** —— 扫描版 PDF 没有文字层，页面上只有一张整页扫描图，
    想拿到「页里的字」只能把整页当图送出去。

    返回 ([(页码, jpeg 字节, mime)], 警告)。渲染不了（缺 PyMuPDF、页码越界）
    不抛异常，通过警告透出 —— 上层据此决定回退到「请提供文本版」的提示。
    """
    try:
        import pymupdf  # PyMuPDF
    except ImportError:  # pragma: no cover
        return [], ["服务端未安装页面渲染组件（PyMuPDF），无法对扫描页做文字识别"]

    warnings: list[str] = []
    out: list[tuple[int, bytes, str]] = []
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            total = doc.page_count
            for no in page_nos:
                if no < 1 or no > total:
                    warnings.append(f"第 {no} 页超出文档页数（共 {total} 页），已跳过")
                    continue
                page = doc[no - 1]
                pix = page.get_pixmap(dpi=dpi)
                jpeg = pix.tobytes(output="jpg", jpg_quality=82)
                out.append((no, jpeg, "image/jpeg"))
    except Exception as exc:  # noqa: BLE001 - 渲染失败不该让整篇文档 500
        return out, [f"扫描页渲染失败：{type(exc).__name__}: {exc}"]
    return out, warnings


def _from_markdown(text: str) -> tuple[list[tuple[bytes, str, str]], list[str]]:
    """md：图片可能是本地路径，也可能是外链。本地路径一律放弃 —— 服务是另一台机器。"""
    urls: list[tuple[str, str]] = []
    for alt, url in _MD_IMAGE.findall(text):
        low = url.lower()
        if low.startswith(("http://", "https://")):
            urls.append((url, alt))

    if not urls:
        return [], []

    import httpx

    out: list[tuple[bytes, str, str]] = []
    warnings: list[str] = []
    failed = 0
    try:
        with httpx.Client(timeout=REMOTE_TIMEOUT, follow_redirects=True) as client:
            for url, alt in urls:
                try:
                    resp = client.get(url)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    warnings.append(f"图片外链无法下载，已跳过：{url[:80]}（{type(exc).__name__}）")
                    continue
                if resp.status_code != 200:
                    failed += 1
                    warnings.append(f"图片外链返回 HTTP {resp.status_code}，已跳过：{url[:80]}")
                    continue
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                ext = ctype.split("/")[-1] if ctype.startswith("image/") else "png"
                if ext not in _MIME_BY_EXT:
                    ext = "jpeg" if resp.content[:2] == b"\xff\xd8" else "png"
                if len(resp.content) > MAX_REMOTE_BYTES:
                    failed += 1
                    warnings.append(f"图片超过 {MAX_REMOTE_BYTES // 1024 // 1024}MB，已跳过：{url[:80]}")
                    continue
                out.append((resp.content, ext, alt))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"图片外链下载失败：{exc}")

    if failed and not out:
        warnings.append("文档中的所有图片外链都无法获取，图中内容未能入库")
    return out, warnings


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
def extract(ext: str, data: bytes, *, text_for_links: str = "") -> ExtractResult:
    """按格式抽出图片。**同步 CPU 操作**，由调用方丢到线程池。

    text_for_links 只给 md 用（要从中解析 `![](url)`）。
    """
    result = ExtractResult()
    ext = (ext or "").lower()

    if ext == ".docx":
        raw, warn = _from_docx(data)
        raw_all: list[tuple[bytes, str, object]] = [(b, e, rid) for b, e, rid in raw]
        result.warnings.extend(warn)
        page_or_alt = "rid"
    elif ext == ".pdf":
        raw_pdf, warn = _from_pdf(data)
        raw_all = [(b, e, pg) for b, e, pg in raw_pdf]
        result.warnings.extend(warn)
        page_or_alt = "page"
    elif ext == ".md":
        try:
            text = text_for_links or data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = text_for_links
        raw_md, warn = _from_markdown(text)
        raw_all = [(b, e, alt) for b, e, alt in raw_md]
        result.warnings.extend(warn)
        page_or_alt = "alt"
    else:
        return result

    if raw_all:
        result.saw_image_marker = True

    seen: set[str] = set()
    skipped = 0
    for blob, iext, extra in raw_all:
        digest = hashlib.sha1(blob).hexdigest()
        if digest in seen:
            # 同一张图被反复引用（页眉 logo、产品手册里重用的同一张示意图），
            # 识别两遍是双倍开销却没有新信息，描述还会重复占位。
            skipped += 1
            continue
        if _decorative(blob):
            skipped += 1
            continue
        seen.add(digest)
        seq = len(result.images) + 1
        result.images.append(
            ImageRef(
                seq=seq,
                data=blob,
                mime=mime_for(iext),
                ext=iext,
                source=ext.lstrip("."),
                page=extra if page_or_alt == "page" else None,
                alt=extra if page_or_alt == "alt" else "",
            )
        )

    if skipped:
        result.warnings.append(f"{skipped} 张图片过小或重复，已作为装饰元素跳过")
    return result


def count_image_blocks(blocks: Iterable[str]) -> int:
    """数出 blocks 里有多少个图片占位块 —— 上层用它校验能否安全配对。"""
    return sum(1 for b in blocks if b.lstrip().startswith("【图】"))


def apply_descriptions(
    blocks: list[str],
    images: list[ImageRef],
    descriptions: dict[int, str],
) -> tuple[list[str], int]:
    """把识别出的图内容写回占位块。

    两条路：
    * **就地替换**：占位块数 == 图片数（docx / md）。把识别结果接在占位符后面，
      题注得以保留 —— 「图 2-3 设备安装示意图」本身就是很强的检索锚点，不该丢。
    * **整块追加**：正文里没有占位符（pdf 走纯文本流，图在文本层之外）。
      每张图单独成块，块里带上页码，否则客户问到时完全无从定位。

    为什么数量对不上时**只追加不替换**：占位块与图片一旦错位，
    第 3 张图的内容会被安到第 2 张图的题注下面。这种错误在页面上看不出来，
    但 AI 会理直气壮地用错误的 Asset 描述回答客户。
    """
    placeholders = count_image_blocks(blocks)
    described = {seq: t for seq, t in descriptions.items() if t and t.strip()}

    if placeholders == len(images) and described:
        out: list[str] = []
        seq = 0
        applied = 0
        for block in blocks:
            if block.lstrip().startswith("【图】"):
                seq += 1
                text = described.get(seq)
                out.append(f"{block.rstrip()}\n{text.strip()}" if text else block)
                if text:
                    applied += 1
            else:
                out.append(block)
        return out, applied

    appended = 0
    out = list(blocks)   # 不改调用者的列表：入参在编排层还要复用（取上下文等）
    for ref in images:
        text = described.get(ref.seq)
        if not text:
            continue
        where = f"第 {ref.page} 页插图" if ref.page else f"第 {ref.seq} 张插图"
        out.append(f"【图】{where}\n{text.strip()}")
        appended += 1
    return out, appended


def context_around(blocks: list[str], seq: int, max_chars: int = 400) -> str:
    """取第 seq 个占位块附近的正文，作为给视觉模型的上下文。

    给模型上下文不是为了让它更啰嗦，是为了**防错**：同一张「接口的图」
    在「硬件安装」章节和「故障排查」章节读出来的侧重完全不同，
    有上下文能显著降低它凭印象补内容的概率。
    """
    del max_chars  # 预留给后续按需裁剪，当前仅需占位块前后各一块
    seen = 0
    for i, block in enumerate(blocks):
        if not block.lstrip().startswith("【图】"):
            continue
        seen += 1
        if seen != seq:
            continue
        before = next((b for b in reversed(blocks[:i]) if not b.lstrip().startswith("【图】")), "")
        after = next((b for b in blocks[i + 1:] if not b.lstrip().startswith("【图】")), "")
        return "\n".join(x for x in (before, after) if x).strip()[:800]
    return ""
