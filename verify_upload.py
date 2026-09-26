"""知识库上传能力真机验证（HTTP 全链路）。

覆盖「上传 → 解析 → 切分 → 向量化 → 入库 → 检索召回」整条链路，
外加三类失败分支必须给可操作提示：
  - 扫描件（无文字层）
  - 损坏的 PDF
  - 不在白名单的格式

用法：
    python verify_upload.py            # 需先启动服务，默认打 8000 端口
    AICS_BASE=http://127.0.0.1:8123 python verify_upload.py

退出码非 0 表示有失败项。会自建临时知识库，跑完自动删除。
"""
from __future__ import annotations

import os
import sys

import httpx

BASE = os.environ.get("AICS_BASE", "http://127.0.0.1:8000")
STAR = ("star@demo.local", "demo12345")
KB_NAME = "上传能力验证临时库"

PASS = FAIL = 0
_LINES: list[str] = []


def emit(s: str) -> None:
    _LINES.append(s)


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        emit(f"  [PASS] {name}")
    else:
        FAIL += 1
        emit(f"  [FAIL] {name}  {detail}")


def banner(t: str) -> None:
    emit(f"\n=== {t} ===")


# --------------------------------------------------------------------------- #
# 造文件（不依赖 reportlab / fpdf：手工拼 PDF）
# --------------------------------------------------------------------------- #
def _pdf(objects: list[bytes]) -> bytes:
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


def ascii_pdf(lines: list[str]) -> bytes:
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


def cjk_pdf(lines: list[str]) -> bytes:
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


def empty_pdf() -> bytes:
    """无文字层的 PDF —— 等价于扫描件 / 纯图片页。"""
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            b"<< /Length 0 >>\nstream\nendstream",
        ]
    )


CJK_LINES = [
    "露营帐篷保修与退换政策",
    "",
    "保修：三人露营帐篷提供36个月厂家保修，覆盖缝线开裂与拉链破损。",
    "退换：帐篷支持交货后14天无理由退换，需保持未使用且保留原包装。",
]


def main() -> int:
    c = httpx.Client(base_url=BASE, timeout=90.0, trust_env=False)

    banner("0. 服务与解析组件")
    try:
        h = c.get("/healthz").json()
        check("服务可用", h.get("ok") is True, str(h)[:120])
    except Exception as exc:  # noqa: BLE001
        check("服务可用", False, f"{exc}（先启动服务，或设 AICS_BASE）")
        banner("结果")
        emit(f"  通过 {PASS} 项，失败 {FAIL} 项")
        return 1

    r = c.post("/api/auth/login", json={"email": STAR[0], "password": STAR[1]})
    check("租户管理员登录", r.status_code == 200, r.text[:160])
    auth = {"Authorization": f"Bearer {r.json()['access_token']}"}

    kb = c.post("/api/knowledge-bases", headers=auth, json={"name": KB_NAME, "description": "自动验证"})
    if kb.status_code == 409:
        items = c.get("/api/knowledge-bases", headers=auth).json()["items"]
        kb_id = next(k["id"] for k in items if k["name"] == KB_NAME)
    else:
        check("创建临时知识库", kb.status_code == 200, kb.text[:160])
        kb_id = kb.json()["id"]

    # ------------------------------------------------------------------ #
    banner("1. 中文 PDF（有文字层）")
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[("files", ("露营帐篷政策.pdf", cjk_pdf(CJK_LINES), "application/pdf"))],
    )
    check("上传接口 200", r.status_code == 200, r.text[:200])
    item = (r.json().get("items") or [{}])[0]
    emit(f"     status={item.get('status')} chunks={item.get('chunks')} error={item.get('error')}")
    check("中文 PDF 入库成功", item.get("status") in ("success", "partial"), str(item))
    check("切出了片段", (item.get("chunks") or 0) > 0, str(item))

    docs = c.get(f"/api/knowledge-bases/{kb_id}/documents", headers=auth).json()["items"]
    row = next((d for d in docs if d["filename"] == "露营帐篷政策.pdf"), None)
    check("文档列表可见且 ext=.pdf", bool(row) and row["ext"] == ".pdf", str(row))

    # ------------------------------------------------------------------ #
    banner("2. 内容能否被检索召回（证明文字层真的进来了）")
    for q in ("露营帐篷保修多久", "帐篷可以退换吗"):
        st = c.post(
            "/api/knowledge/search-test", headers=auth, json={"kb_id": kb_id, "query": q, "top_k": 3}
        ).json()
        items = st.get("items") or []
        emit(f"     Q: {q} -> {len(items)} 命中，top={items[0].get('score') if items else '-'}")
        check(f"能召回 PDF 内容：{q}", bool(items), str(st)[:150])
        if items:
            emit(f"        {items[0].get('text', '')[:70]}")

    # ------------------------------------------------------------------ #
    banner("3. 英文 PDF")
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[("files", ("tent_en.pdf", ascii_pdf(["Tent Warranty 36 months", "Return within 14 days"]), "application/pdf"))],
    )
    item = (r.json().get("items") or [{}])[0]
    check("英文 PDF 入库成功", item.get("status") in ("success", "partial"), str(item))

    # ------------------------------------------------------------------ #
    banner("4. 扫描件（无文字层）必须给可操作提示")
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[("files", ("scanned.pdf", empty_pdf(), "application/pdf"))],
    )
    item = (r.json().get("items") or [{}])[0]
    err = item.get("error") or ""
    emit(f"     status={item.get('status')} error={err[:130]}")
    check("无文字层判定为失败", item.get("status") == "failed", str(item))
    check("说明是扫描件/无文字层", "文字层" in err or "扫描" in err, err)
    check("给出替代方案", "DOCX" in err or "TXT" in err, err)

    # ------------------------------------------------------------------ #
    banner("5. 损坏的 PDF 不能拖垮整个请求")
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[("files", ("broken.pdf", b"%PDF-1.4 this is not a real pdf", "application/pdf"))],
    )
    check("返回 200 且单文件失败", r.status_code == 200, f"{r.status_code} {r.text[:160]}")
    item = (r.json().get("items") or [{}])[0]
    emit(f"     status={item.get('status')} error={(item.get('error') or '')[:130]}")
    check("坏文件标记为失败", item.get("status") == "failed", str(item))

    # ------------------------------------------------------------------ #
    banner("6. 白名单外格式")
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[("files", ("archive.zip", b"PK\x03\x04", "application/zip"))],
    )
    item = (r.json().get("items") or [{}])[0]
    err = item.get("error") or ""
    emit(f"     error={err[:130]}")
    check("拒绝并列出支持格式", item.get("status") == "failed" and ".pdf" in err, err)

    # 混合上传：一个坏一个坏不应影响好文件
    r = c.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files=[
            ("files", ("ok.pdf", cjk_pdf(["退货政策：七天无理由退货"]), "application/pdf")),
            ("files", ("bad.xyz", b"junk", "application/octet-stream")),
        ],
    )
    data = r.json()
    ok_item = next((i for i in data.get("items", []) if i["filename"] == "ok.pdf"), {})
    check("混合上传时好文件仍然成功", ok_item.get("status") in ("success", "partial"), str(data)[:200])
    check("好文件被计入 accepted", data.get("accepted") == 1, str(data.get("accepted")))

    # ------------------------------------------------------------------ #
    banner("7. 清理")
    d = c.delete(f"/api/knowledge-bases/{kb_id}", headers=auth)
    check("临时知识库已删除", d.status_code == 200, str(d.status_code))
    left = c.get("/api/knowledge-bases", headers=auth).json()["items"]
    check("列表里已看不到临时库", all(k["name"] != KB_NAME for k in left), str([k["name"] for k in left]))

    banner("结果")
    emit(f"  通过 {PASS} 项，失败 {FAIL} 项")
    return 1 if FAIL else 0


def emit_report() -> None:
    """把结果一并打印出来（走 stdout，和其他 verify_*.py 保持一致）。"""
    print("\n".join(_LINES))


if __name__ == "__main__":
    code = main()
    emit_report()
    sys.exit(code)
