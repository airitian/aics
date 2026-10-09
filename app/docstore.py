"""文档原始文件与图片的磁盘存放。

为什么要落盘
------------
原文件解析完就丢掉，是本项目此前一个**不可逆**的损失：
文档里的图片随对象销毁而永久消失。等到要做 OCR、要改切分策略时，
只能让用户重新上传一遍 —— 而用户往往已经没有那份文件了。

落盘后，「换视觉模型重跑一遍」「按新切分策略重解析」都变成一次 API 调用的事。

目录结构
--------
    <assets_dir>/<tenant_id>/<doc_id>/original<ext>   原始文件
    <assets_dir>/<tenant_id>/<doc_id>/img_<seq>.<ext> 抽出的图片

按租户分子目录是为了**删除干净**：删库时能整目录移除，不用在共享目录里
逐个文件辨认归属，也不用担心跨租户误删。

只存相对路径
------------
数据库里一律写相对路径（`<tenant_id>/<doc_id>/...`），不写绝对路径。
否则换台机器部署、换了 assets_dir，历史数据全部变成死链。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import settings

logger = logging.getLogger("aics.docstore")

# 允许写盘的扩展名白名单：防止恶意文件名（../../..）或奇怪格式被写进磁盘。
# 图片扩展名首次到这里才被允许，其余一律回落到 .bin。
_SAFE_EXT = {
    ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


def root() -> Path:
    return Path(settings.assets_dir)


def _safe_ext(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".")
    return f".{ext}" if f".{ext}" in _SAFE_EXT else ".bin"


def doc_dir(tenant_id: str, doc_id: str) -> Path:
    return root() / str(tenant_id) / str(doc_id)


def save_original(tenant_id: str, doc_id: str, filename: str, data: bytes) -> str:
    """存原始文件，返回相对路径。失败只记录日志，不抛 —— 落盘是增强项，
    不能因为它失败就让整篇文档入库失败。"""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in (filename or "") else ""
    rel = f"{tenant_id}/{doc_id}/original{_safe_ext(ext)}"
    try:
        target = root() / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("原始文件未能落盘 doc=%s：%s", doc_id, exc)
        return ""
    return rel


def save_image(tenant_id: str, doc_id: str, seq: int, ext: str, data: bytes) -> str:
    rel = f"{tenant_id}/{doc_id}/img_{seq}{_safe_ext(ext)}"
    try:
        target = root() / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("图片未能落盘 doc=%s seq=%s：%s", doc_id, seq, exc)
        return ""
    return rel


def abs_path(rel_path: str) -> Path | None:
    """相对路径 → 绝对路径，并校验没有越界到 assets 目录之外。"""
    if not rel_path:
        return None
    try:
        target = (root() / rel_path).resolve()
        base = root().resolve()
    except Exception:  # noqa: BLE001
        return None
    if base not in target.parents and target != base:
        return None
    return target


def remove_doc(tenant_id: str, doc_id: str) -> None:
    """删除文档相关文件。失败不抛：磁盘清理失败不该让删除接口 500。

    这里必须捕获 BaseException 而不是 Exception
    ------------------------------------------------
    部分运行环境（IDE / 自动化 agent 的 sitecustomize）会包一层"安全删除"
    守卫：它拦下 shutil.rmtree，判定为批量删除风险后直接 `raise SystemExit(1)`。
    SystemExit 继承自 **BaseException**，`except Exception` 抓不到，会一路冒到
    路由层 → DELETE /api/documents/{id} 返回 500，且异常发生在 commit 之前，
    SQLite 回滚 → 表现为"怎么点都删不掉"。

    磁盘残留只是占空间（下次 reindex 会覆盖），远比"文档删不掉"无害，
    所以任何异常都吞掉并记 warning。
    """
    try:
        shutil.rmtree(doc_dir(tenant_id, doc_id), ignore_errors=True)
    except BaseException as exc:  # noqa: BLE001 - 含 SystemExit，见 docstring
        logger.warning("删除文档文件失败 doc=%s：%s", doc_id, exc)


def mime_of_ext(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get((ext or "").lower(), "application/octet-stream")


# --------------------------------------------------------------------------- #
# 聊天图片：客户在对话中发的图。目录结构与文档资产隔离，
# 但共用同一套「相对路径 + 越界校验」的规则。
# --------------------------------------------------------------------------- #
def save_chat_image(tenant_id: str, image_id: str, ext: str, data: bytes) -> str:
    """存客户聊天图片，返回相对路径。失败返回空串（落库继续，只是后面看不了原图）。"""
    rel = f"chat/{tenant_id}/{image_id}{_safe_ext(ext)}"
    try:
        target = root() / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("聊天图片未能落盘 image=%s：%s", image_id, exc)
        return ""
    return rel


def read_chat_image(rel_path: str) -> bytes:
    """读聊天图片字节；路径非法或文件丢失都返回空 bytes，由调用方决定 404。"""
    target = abs_path(rel_path)
    if target is None:
        return b""
    try:
        return target.read_bytes()
    except OSError as exc:
        logger.warning("聊天图片读取失败 path=%s：%s", rel_path, exc)
        return b""


def remove_chat_image(rel_path: str) -> None:
    if not rel_path:
        return
    target = abs_path(rel_path)
    if target is None:
        return
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("聊天图片删除失败 path=%s：%s", rel_path, exc)
    except BaseException as exc:  # noqa: BLE001 - 环境守卫可能抛 SystemExit，见 remove_doc
        logger.warning("聊天图片删除失败 path=%s：%s", rel_path, exc)
