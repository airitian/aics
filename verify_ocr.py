"""OCR / 视觉模型横评 —— 决定「文档里的图要不要识别、用哪个模型」时用。

用法：
    python verify_ocr.py                    # 用内置生成的转置参数表跑全部候选
    python verify_ocr.py --image a.png      # 换成你自己的图
    python verify_ocr.py --models Qwen3-VL-32B-Instruct,PaddleOCR-VL-1.5

判据不是「好不好看」，是三条硬指标：
1. **关键值命中率** —— 图里的型号/重量/价格有没有被完整取出（漏一个数字就是错参数）
2. **两次一致性** —— 同一张图跑两遍结果是否相同（不同 = 模型在猜，不可信）
3. **表格能否闭环** —— OCR 结果能否被 tablelayout 判为转置表并按型号切块

**结论必须靠这三条，不能靠肉眼看一次输出。**
实测过：同一个模型在干净的图上全对，在带框线的表格图上会凭空编造；
另一个模型输出看着完整，却把 MTK7621+7612 认成 MTK7621+761（少一位）。
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import httpx  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None

API_BASE = (os.getenv("EMBED_BASE_URL") or "https://ai.gitee.com/api/v1").rstrip("/")
API_KEY = os.getenv("EMBED_API_KEY") or ""

# 候选模型：先 OCR 专用，再通用视觉，最后大模型兜底
DEFAULT_MODELS = [
    "PaddleOCR-VL-1.5",
    "Unlimited-OCR",
    "Qwen3-VL-32B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-plus",
    "InternVL3-38B",
]

PROMPT = "把图中表格还原为 markdown 表格，保留全部行列与原文，不要翻译不要省略。只输出表格。"

# 内置测试图：转置参数表（列 = 型号，行 = 属性）—— 产品手册最常见的坑型
_TEST_GRID = [
    ["AI Smart Bridge", "Outdoor 5G Series", "Outdoor 5G Series", "5G/450M Series"],
    ["Model", "WB730", "WB620E", "WB620F"],
    ["Chipset", "MTK7621+7612", "MTK7621+7612", "MTK7620+7612E"],
    ["Flash/RAM", "8MB/64MB", "8MB/64MB", "8MB/64MB"],
    ["PTP Distance", "5KM", "15KM", "10KM"],
    ["CTN weight", "15.5KG", "13KG", "11.5KG"],
    ["PRICE(RMB)", "<80PCS,Y165", "<100PCS,Y145", "<100PCS,Y145"],
]
# 关键值：漏任何一个，下游都可能给客户报错参数
_TRUTH = [
    "WB730", "WB620E", "WB620F",
    "MTK7621+7612", "MTK7620+7612E",
    "15.5KG", "13KG", "11.5KG",
    "5KM", "15KM", "10KM",
    "8MB/64MB",
]


def build_test_image() -> bytes:
    if Image is None:
        raise SystemExit("需要 Pillow：pip install pillow")
    font = None
    for p in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, 22)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    cw, ch = 155, 46
    W, H = cw * 4 + 20, ch * len(_TEST_GRID) + 20
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for r, row in enumerate(_TEST_GRID):
        for c, v in enumerate(row):
            x, y = 10 + c * cw, 10 + r * ch
            d.rectangle([x, y, x + cw, y + ch], outline="#999")
            d.text((x + 6, y + 10), v, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def call_vl(model: str, b64: str, prompt: str = PROMPT, max_tokens: int = 2000) -> tuple[str, float, dict]:
    """返回 (文本, 耗时ms, usage)。失败返回 ("", ms, {})，错误原因打到 stderr。"""
    t = time.time()
    try:
        r = httpx.post(
            API_BASE + "/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=180,
        )
    except Exception as exc:
        print(f"    EXC {type(exc).__name__}: {str(exc)[:100]}", file=sys.stderr)
        return "", (time.time() - t) * 1000, {}
    ms = (time.time() - t) * 1000
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:160]}", file=sys.stderr)
        return "", ms, {}
    data = r.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return content or "", ms, data.get("usage") or {}


def strip_fences(md: str) -> str:
    return md.replace("```markdown", "").replace("```", "").strip()


def to_grid(md: str) -> list[list[str]]:
    """markdown 表格 → 网格。列数取最大行，不足补空（OCR 常漏列）。"""
    rows = []
    for ln in md.split("\n"):
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):  # 分隔线
            continue
        rows.append(cells)
    if not rows:
        return []
    w = max(len(r) for r in rows)
    return [r + [""] * (w - len(r)) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="自定义图片路径（默认用内置生成的参数表）")
    ap.add_argument("--models", help="逗号分隔的模型名")
    ap.add_argument("--consistency", action="store_true", help="同一模型跑两遍，检查一致性")
    ap.add_argument("--closure", action="store_true", help="把 OCR 结果喂给 tablelayout 验证闭环")
    args = ap.parse_args()

    if not API_KEY:
        print("缺少 EMBED_API_KEY（.env 里配置）")
        return 2

    data = Path(args.image).read_bytes() if args.image else build_test_image()
    b64 = "data:image/png;base64," + base64.b64encode(data).decode()
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()] or DEFAULT_MODELS
    truth = _TRUTH if not args.image else None

    print(f"图片 {len(data)} 字节 / 模型 {len(models)} 个\n")

    results: dict[str, str] = {}
    for m in models:
        md, ms, usage = call_vl(m, b64)
        md = strip_fences(md)
        results[m] = md
        if truth:
            hit = sum(1 for v in truth if v in md)
            rate = hit / len(truth)
            flag = "OK  " if rate >= 0.95 else ("可疑" if rate >= 0.7 else "差  ")
            print(f"{flag} {m:26s} {ms:6.0f}ms  关键值 {hit}/{len(truth)} ({rate:.0%})  "
                  f"tokens={usage.get('total_tokens', '-')}")
            miss = [v for v in truth if v not in md]
            if miss:
                print(f"      漏/错: {miss}")
        else:
            print(f"    {m:26s} {ms:6.0f}ms  {len(md)} 字")

        if args.consistency and md:
            md2, _, _ = call_vl(m, b64)
            md2 = strip_fences(md2)
            same = md == md2
            print(f"      二次一致性: {'一致' if same else '不一致 ← 模型在猜，不可信'}")

    if args.closure:
        from app.tablelayout import parse_sheet

        print("\n=== 表格闭环（OCR → tablelayout）===")
        for m, md in results.items():
            grid = to_grid(md)
            if not grid:
                print(f"  {m}: 输出不是表格，跳过")
                continue
            pieces, _ = parse_sheet(grid, sheet_name="Sheet1", max_chars=600)
            print(f"  {m}: {len(grid)}行 x {len(grid[0])}列 → {len(pieces)} 块")
            for p in pieces[:2]:
                print(f"      {p[:110]}")

    print("\n判读：关键值 <95% 或二次不一致的模型，都不该把结果直接入库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
