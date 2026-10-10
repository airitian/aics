"""知识库：知识库管理、文档上传与解析、检索测试。"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app import audit, docimage, docstore, rag, vision
from app.config import settings
from app.database import SessionLocal, get_db
from app.deps import client_ip, get_scope, get_tenant, require_roles
from app.models import (
    AiEmployee,
    AssetStatus,
    Chunk,
    DocStatus,
    Document,
    DocumentAsset,
    KnowledgeBase,
    Tenant,
    User,
    UserRole,
)
from app.rag import purge_document, purge_kb
from app.scoping import TenantScope, add_scoped
from app.schemas import KbIn, SearchTestIn
from app.textparse import ParseError, parse_document
from app.utils import new_id

logger = logging.getLogger("aics.knowledge")

router = APIRouter(prefix="/api", tags=["knowledge"])
editor = require_roles(UserRole.TENANT_ADMIN, UserRole.CONFIG_EDITOR)


# --------------------------------------------------------------------------- #
# 短写事务
#
# 为什么入库链路必须自己管事务，而不是沿用请求作用域那个 session：
#
# SQLite 是**单写者**模型，`PRAGMA busy_timeout=8000`（8 秒）只能兜住抖动，
# 兜不住"锁被持有几分钟"。而入库链路上有三处远程调用，每处都可能耗时几十秒：
#   1. vision.describe_many   —— 逐图调视觉模型
#   2. embed_texts            —— 按 EMBED_BATCH 分批，实测每批 1~30 秒 + 3 次重试
#   3. store.upsert           —— 向量库（Qdrant Cloud 在公网）
#
# 只要这些慢调用被夹在一个未提交的写事务中间，锁就一直被这个请求独占，
# 期间任何并发写（另一个上传、员工 PATCH、审计落库）都会在 8 秒后
# 抛 `sqlite3.OperationalError: database is locked` → 路由层 500。
#
# 所以这里的规矩是：**慢调用一律在事务外跑，写入各自用独立短事务。**
# 写锁的持有时间从"分钟"降到"毫秒"，并发写自然就进不进来了。
# --------------------------------------------------------------------------- #
@contextmanager
def short_write() -> Iterator[Session]:
    """独立会话的短写事务：正常结束即提交，异常回滚，**绝不跨越慢调用**。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _insert_document_row(
    *,
    tenant_id: str,
    kb_id: str,
    filename: str,
    ext: str,
    size: int,
) -> tuple[str | None, str]:
    """校验并落 Document 行（status=processing），**全程在一个短事务内**。

    去重 / 文档数上限 / 容量上限三项校验搬进事务里一起判，有两个原因：
    1. 它们依赖"上一个文件已经落库"才准确。拆成独立事务后如果沿用请求开始时
       的快照，同一次请求里传两个同名文件就会双双通过校验、重复入库。
    2. 顺带把「读-判-写」变成原子的，消掉并发上传的 TOCTOU：
       两个请求同时判断"没有同名文档"然后各自 INSERT 的竞态。
       去重靠的是 (kb_id, filename, size_bytes)，没有唯一索引兜底。

    Returns:
        ``(doc_id, "")`` 表示已落库；``(None, 错误文案)`` 表示被拒收。
    """
    with short_write() as db:
        duplicated = db.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id,
                Document.kb_id == kb_id,
                Document.filename == filename[:255],
                Document.size_bytes == size,
            )
        ).scalar_one_or_none()
        if duplicated:
            return None, (
                "该文件已在库中（同名且大小相同），未重复入库；"
                "如需更新请先删除原文档，或对原文档执行「重建索引」"
            )

        # 每次重新数：本请求前面几个文件已经在自己的短事务里提交了，
        # 请求开始时的快照到这一刻已经过期。
        current_docs = int(
            db.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.tenant_id == tenant_id, Document.kb_id == kb_id)
            ).scalar_one()
        )
        if current_docs >= settings.max_docs_per_kb:
            return None, f"该知识库文档数已达上限（{settings.max_docs_per_kb}）"

        # 容量同理：kb_row 是本事务内重读的，已含前面几个文件的增量。
        kb_row = db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant_id
            )
        ).scalar_one_or_none()
        if kb_row is None:
            return None, "知识库不存在"
        if kb_row.total_bytes + size > settings.max_kb_bytes:
            return None, (
                f"知识库容量将超过上限 {settings.max_kb_bytes // 1024 // 1024 // 1024}GB"
            )

        doc = Document(
            id=new_id(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            filename=filename[:255],
            ext=ext,
            size_bytes=size,
            status=DocStatus.PROCESSING,
        )
        db.add(doc)
        # 容量随文档一起累加：每个文件一个短事务，而不是全部文件攒到最后。
        # 这样中途崩溃时已入库的文档仍然计入容量，不会出现"占了空间却没记账"。
        kb_row.total_bytes = max(0, kb_row.total_bytes + size)
        return doc.id, ""


def _finish_document(
    *,
    tenant_id: str,
    doc_id: str,
    status: str,
    chunk_count: int,
    error: str,
) -> None:
    """回写文档最终状态：独立短事务。

    按 id 取对象时**必须带tenant_id**（fail-closed）——
    这已经是独立会话，不再有 TenantScope 的兜底，漏掉租户条件就是越权写。
    """
    with short_write() as db:
        row = db.execute(
            select(Document).where(Document.id == doc_id, Document.tenant_id == tenant_id)
        ).scalar_one_or_none()
        if row is None:
            # 文档在处理期间被删了：无处可写，直接返回（调用方照常记结果）
            return
        row.status = status
        row.chunk_count = chunk_count
        row.error = error


def _kb_out(db: Session, kb: KnowledgeBase) -> dict:
    docs = int(
        db.execute(
            select(func.count()).select_from(Document).where(
                Document.tenant_id == kb.tenant_id, Document.kb_id == kb.id
            )
        ).scalar_one()
    )
    chunks = int(
        db.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.tenant_id == kb.tenant_id, Chunk.kb_id == kb.id
            )
        ).scalar_one()
    )
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "doc_count": docs,
        "chunk_count": chunks,
        "total_bytes": kb.total_bytes,
        "created_at": kb.created_at.isoformat(),
    }


@router.get("/knowledge-bases")
def list_kbs(scope: TenantScope = Depends(get_scope), user: User = Depends(editor)) -> dict:
    rows = scope.list(KnowledgeBase)
    rows.sort(key=lambda k: k.created_at)
    return {"items": [_kb_out(scope.db, k) for k in rows], "max": settings.max_kb_per_tenant}


@router.post("/knowledge-bases")
def create_kb(
    payload: KbIn,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    if scope.count(KnowledgeBase) >= settings.max_kb_per_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"单租户知识库数量已达上限（{settings.max_kb_per_tenant}）",
        )
    dup = scope.db.execute(
        select(KnowledgeBase.id).where(
            KnowledgeBase.tenant_id == tenant.id, KnowledgeBase.name == payload.name.strip()
        )
    ).scalar_one_or_none()
    if dup:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="同名知识库已存在")
    kb = KnowledgeBase(
        id=new_id(), tenant_id=tenant.id, name=payload.name.strip(), description=payload.description
    )
    scope.add(kb)
    scope.db.commit()
    audit.record(scope.db, action="kb.create", tenant_id=tenant.id, target=kb.id,
                 ip=client_ip(request), detail={"name": kb.name}, commit=True)
    return _kb_out(scope.db, kb)


@router.delete("/knowledge-bases/{kb_id}")
def delete_kb(
    kb_id: str,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    kb = scope.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    from app.models import EmployeeKbBinding

    bound = int(
        scope.db.execute(
            select(func.count()).select_from(EmployeeKbBinding).where(
                EmployeeKbBinding.tenant_id == tenant.id, EmployeeKbBinding.kb_id == kb_id
            )
        ).scalar_one()
    )
    if bound:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"该知识库已被 {bound} 个 AI 员工绑定，请先解除绑定",
        )
    purge_kb(scope.db, tenant.id, kb_id)
    for doc in scope.list(Document, kb_id=kb_id):
        docstore.remove_doc(tenant.id, doc.id)
        scope.db.delete(doc)
    # 资产行没有对 documents 的外键约束（按 doc_id 弱关联），必须显式清理，
    # 否则文档删了、图片记录还在，列表里出现「幽灵图片」
    for asset in scope.list(DocumentAsset, kb_id=kb_id):
        scope.db.delete(asset)
    scope.db.delete(kb)
    scope.db.commit()
    audit.record(scope.db, action="kb.delete", tenant_id=tenant.id, target=kb_id,
                 ip=client_ip(request), commit=True)
    return {"ok": True}


@router.get("/documents")
def list_tenant_docs(scope: TenantScope = Depends(get_scope), user: User = Depends(editor)) -> dict:
    """租户级文件清单（跨知识库平铺），供员工编辑器「选择文件」弹窗使用。"""
    rows = scope.list(Document)
    kb_names = {k.id: k.name for k in scope.list(KnowledgeBase)}
    rows.sort(key=lambda d: d.created_at, reverse=True)
    return {
        "items": [
            {
                "id": d.id,
                "kb_id": d.kb_id,
                "kb_name": kb_names.get(d.kb_id, ""),
                "filename": d.filename,
                "ext": d.ext,
                "size_bytes": d.size_bytes,
                "status": d.status,
                "chunk_count": d.chunk_count,
                "enabled": d.enabled,
                "created_at": d.created_at.isoformat(),
            }
            for d in rows
        ]
    }


@router.get("/knowledge-bases/{kb_id}/documents")
def list_docs(
    kb_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    kb = scope.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    rows = scope.list(Document, kb_id=kb_id)
    rows.sort(key=lambda d: d.created_at, reverse=True)
    return {
        "items": [
            {
                "id": d.id,
                "kb_id": d.kb_id,
                "filename": d.filename,
                "ext": d.ext,
                "size_bytes": d.size_bytes,
                "status": d.status,
                "error": d.error,
                "chunk_count": d.chunk_count,
                "enabled": d.enabled,
                "created_at": d.created_at.isoformat(),
            }
            for d in rows
        ],
        "limits": {
            "max_file_bytes": settings.max_upload_bytes,
            "max_files_per_upload": settings.max_upload_files,
            "max_docs_per_kb": settings.max_docs_per_kb,
            "max_kb_bytes": settings.max_kb_bytes,
            "allowed_ext": sorted(settings.allowed_ext),
        },
    }


# --------------------------------------------------------------------------- #
# 文档状态分级：只有「影响内容完整性/可信度」的 warnings 才把文档压成 partial，
# 纯处理说明（页眉剥离、按设计跳过的标识符片段等）照常展示但不改变状态。
#
# 为什么用正则白名单而不是给每处生成点打机器标记：warnings 文本会原样展示给用户
# （admin 界面/上传响应），不想让它携带 [INFO] 之类的内部标签；
# 新增生成点默认按「影响状态」处理 —— 宁可多标（退回旧行为），不能漏标。
# --------------------------------------------------------------------------- #
_INFO_WARNING_RES = tuple(
    re.compile(p)
    for p in (
        r"^已剥离 .+页眉/页脚行$",                        # 结构清洗，内容无损失
        r"标识符片段的向量化",                           # 按设计仅支持精确号码查询
        r"图片已识别为文字并入库",                       # 图片识别成功是好消息
        r"图片来自已转写的扫描页，整页文字已入库，不再重复识别",
        r"图片(过小或重复，已作为装饰元素跳过|已作为装饰元素跳过)",
        r"合并单元格，已向下填充",                       # 修复性处理，内容更完整
        r"分组表头.*已按多行表头合并",
        r"已保守按行切分",                               # 表格形态保守降级，内容未丢
        r"已按逗号处理",
        r"已按纯段落处理",
    )
)


def _degrades_doc_status(warnings: list[str]) -> bool:
    """warnings 里只要有一条不在「纯处理说明」白名单内 → partial。"""
    for w in warnings:
        if not any(p.search(w) for p in _INFO_WARNING_RES):
            return True
    return False


async def _handle_document_images(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    filename: str,
    ext: str,
    data: bytes,
    doc_text: str,
    blocks: list[str],
    ocr_pages: set[int] | None = None,
) -> tuple[list[str], list[str]]:
    """抽出文档里的图，并在启用时把它读成文字写回占位块。

    这条链路的每一环都可能失败：原文件落不了盘、图片抽不出来、视觉模型超时。
    任何一种失败都**不能让整篇文档入库失败** ——
    丢一张图的内容只是少一块知识，整篇文档失败则是用户白传一次。

    ocr_pages：解析层已做整页转写的扫描页页码。这些页的内嵌图就是整页扫描图，
    内容已经抄成文字了，必须跳过 —— 否则同一页内容入库两遍，还各占一个配额。

    **事务边界**：这里有两处写入（图片元数据落库、识别结果回写），两次之间夹着
    ``vision.describe_many`` —— 逐图调视觉模型，是整条链路上最慢的一环。
    所以两次写入各自用一个独立短事务（``short_write``），绝不让视觉调用发生在
    未提交的事务中间。原来这里直接用请求作用域的 session，等于把视觉模型的
    耗时全算进写锁里。
    """
    warnings: list[str] = []
    ocr_pages = ocr_pages or set()
    # 原文件必须落盘：解析完就丢掉的话，换视觉模型重跑、按新策略重解析就没机会了
    await run_in_threadpool(docstore.save_original, tenant_id, doc_id, filename, data)

    try:
        ex = await run_in_threadpool(
            docimage.extract, ext, data, **{"text_for_links": doc_text}
        )
    except Exception as exc:  # noqa: BLE001 - 抽图失败不该连累正文
        logger.warning("图片抽出失败 doc=%s：%s", doc_id, exc)
        return blocks, warnings
    if ocr_pages:
        skipped_scan = [r for r in ex.images if r.page in ocr_pages]
        if skipped_scan:
            ex.images = [r for r in ex.images if r.page not in ocr_pages]
            ex.saw_image_marker = bool(ex.images)
            warnings.append(
                f"{len(skipped_scan)} 张图片来自已转写的扫描页，整页文字已入库，不再重复识别"
            )
    warnings.extend(ex.warnings)

    if not ex.images:
        if ex.saw_image_marker:
            warnings.append("文档中的图片已作为装饰元素跳过")
        return blocks, warnings

    assets: list[DocumentAsset] = []
    for ref in ex.images:
        rel = await run_in_threadpool(
            docstore.save_image, tenant_id, doc_id, ref.seq, ref.ext, ref.data
        )
        width, height = docimage.image_size(ref.data)
        asset = DocumentAsset(
            id=new_id(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            doc_id=doc_id,
            seq=ref.seq,
            source=ref.source,
            mime=ref.mime,
            ext=ref.ext,
            size_bytes=len(ref.data),
            width=width,
            height=height,
            page_index=ref.page or 0,
            rel_path=rel,
            status=AssetStatus.SKIPPED,
        )
        assets.append(asset)

    # 短事务 1：图片元数据落库后立刻提交释放写锁，再去做慢的视觉识别。
    # add_scoped 保留了原来的归属校验（对象 tenant_id 与作用域不符则拒绝写入）。
    with short_write() as db:
        for asset in assets:
            add_scoped(db, asset, tenant_id)

    total = len(assets)
    if not vision.enabled():
        warnings.append(
            f"文档含 {total} 张图片，当前未启用图片识别，图中内容未能入库。"
            f"AI 遇到图里的信息会如实说读不到，而不是猜。"
        )
        return blocks, warnings

    capped = ex.images[: max(0, settings.vision_max_images)]
    items = [(r.seq, r.data, r.mime, docimage.context_around(blocks, r.seq)) for r in capped]
    # ↓ 慢调用：此处不持有任何写事务
    descriptions = await vision.describe_many(items)

    # 短事务 2：识别结果回写。按 (doc_id, seq) 定位而不是用上面那些 ORM 对象——
    # 它们属于已关闭的会话（expire 之后再访问属性会触发刷新查询）。
    by_seq = {a.seq: a.id for a in assets}
    rows = [
        {"id": by_seq[seq], "description": text}
        for seq, text in descriptions.items()
        if seq in by_seq
    ]
    if rows:
        with short_write() as db:
            for row in rows:
                db.execute(
                    update(DocumentAsset)
                    .where(
                        DocumentAsset.id == row["id"],
                        DocumentAsset.tenant_id == tenant_id,
                    )
                    .values(description=row["description"], status=AssetStatus.OK)
                )

    blocks, applied = docimage.apply_descriptions(blocks, ex.images, descriptions)
    if applied:
        warnings.append(f"{applied} 张图片已识别为文字并入库")
    if total - applied:
        warnings.append(
            f"{total - applied} 张图片未能识别，AI 遇到这部分内容会如实说读不到"
        )
    return blocks, warnings


@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_docs(
    kb_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    """上传文档。**逐文件独立事务**：慢调用一律在写事务之外。

    事务结构（这是本函数存在的全部理由，见``short_write`` 的注释）：

    ========== ==================================== ==============
    阶段        动作                                  是否持写锁
    ========== ==================================== ==============
    A校验      扩展名 / 大小（纯内存）                否
    B 落doc   INSERT Document + 累加 kb.total_bytes   **是（毫秒级）**
    C 解析     parse_document（线程池，CPU +磁盘）    否
    D 图片     落图片元数据 → 视觉识别 → 回写          **是（两次，毫秒级）**
    E 索引     切分 + 向量化 + 写向量库 + 落chunks    **是（一次，毫秒级）**
    F 收尾     回写 doc.status / chunk_count         **是（毫秒级）**
    ========== ==================================== ==============

    原来的写法是「整个 for 循环共用一个事务，循环结束才commit」，
    于是 C/D/E 三个阶段的全部远程调用（合计可达数分钟）都发生在写锁内，
    并发写必然撞 ``database is locked``。
    """
    kb = scope.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"单次最多上传 {settings.max_upload_files} 个文件，本次提交 {len(files)} 个",
        )

    results: list[dict] = []
    accepted_docs = 0

    for upload in files:
        filename = upload.filename or "unnamed"
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        data = await upload.read()

        # 阶段 A：纯内存校验，连数据库都不碰
        if ext not in settings.allowed_ext:
            results.append(
                {
                    "filename": filename,
                    "status": DocStatus.FAILED,
                    "error": f"不支持的格式 {ext or '（无扩展名）'}。支持：{'、'.join(sorted(settings.allowed_ext))}",
                }
            )
            continue
        if len(data) > settings.max_upload_bytes:
            results.append(
                {
                    "filename": filename,
                    "status": DocStatus.FAILED,
                    "error": f"文件超过单文件上限 {settings.max_upload_bytes // 1024 // 1024}MB",
                }
            )
            continue

        # 阶段 B：去重 + 上限校验 + 落 doc 行，一个短事务内完成，随即释放写锁。
        # 去重为什么必须拦：重复副本会产生**同分**的重复片段，把 top_k 召回位占满
        # （实测同一份 37 片段 PDF 传 4 次后，检索 top-5 里 4 条是同一段文字、分数一模一样），
        # 结果既浪费提示词预算，又把其他文档挤出去。
        doc_id, reject_reason = _insert_document_row(
            tenant_id=tenant.id,
            kb_id=kb_id,
            filename=filename,
            ext=ext,
            size=len(data),
        )
        if doc_id is None:
            results.append(
                {"filename": filename, "status": DocStatus.FAILED, "error": reject_reason}
            )
            continue
        accepted_docs += 1

        # 阶段 C~F：全程不持有写事务
        try:
            # 阶段 C：解析（纯 CPU / 磁盘，线程池里跑，本来就不碰 DB）
            parsed = await run_in_threadpool(parse_document, filename, data)
            # 阶段 D：图片元数据 + 视觉识别（内部两次短事务）
            blocks, image_warnings = await _handle_document_images(
                tenant.id, kb_id, doc_id, filename, ext, data,
                parsed.text, list(parsed.blocks), ocr_pages=parsed.ocr_pages,
            )
            # 阶段 E：切分 + 向量化 + 落 chunks。
            # index_document 内部是「先 await 向量模型，再 db.add + flush」，
            # 所以把写事务的开销压到 flush 之后的极短窗口；向量模型那段慢调用
            # 发生在事务**建立之前**，不会占着写锁。
            with short_write() as db:
                index_res = await rag.index_document(
                    db,
                    tenant_id=tenant.id,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    text=parsed.text,
                    blocks=blocks,
                )
            warnings = list(parsed.warnings) + image_warnings + list(index_res.warnings)
            doc_status = (
                DocStatus.PARTIAL if _degrades_doc_status(warnings) else DocStatus.SUCCESS
            )
            # 阶段 F：回写最终状态（独立短事务）
            _finish_document(
                tenant_id=tenant.id,
                doc_id=doc_id,
                status=doc_status,
                chunk_count=index_res.chunk_count,
                error="；".join(warnings)[:2000],
            )
            results.append(
                {
                    "filename": filename,
                    "doc_id": doc_id,
                    "status": doc_status,
                    "chunks": index_res.chunk_count,
                    "warnings": warnings,
                }
            )
        except ParseError as exc:
            _finish_document(
                tenant_id=tenant.id, doc_id=doc_id, status=DocStatus.FAILED,
                chunk_count=0, error=exc.message,
            )
            results.append(
                {"filename": filename, "doc_id": doc_id, "status": DocStatus.FAILED, "error": exc.message}
            )
        except Exception as exc:  # noqa: BLE001 - 单文件失败不能影响其他文件
            logger.exception("文档入库失败 filename=%s", filename)
            err = f"处理失败：{type(exc).__name__}: {exc}"
            # 失败也必须回写状态：否则文档会永远停在 processing，
            # 用户在前端看到的是"一直在处理中"，既不知道成没成也没法重试。
            _finish_document(
                tenant_id=tenant.id, doc_id=doc_id, status=DocStatus.FAILED,
                chunk_count=0, error=err,
            )
            results.append(
                {"filename": filename, "doc_id": doc_id, "status": DocStatus.FAILED, "error": err}
            )

    # 审计：独立短事务。审计是安全留痕，绝不能因为它失败而把上面的入库结果带崩，
    # 也不能让它去join 一个横跨数分钟的旧事务。
    try:
        with short_write() as db:
            audit.record(
                db,
                action="kb.upload",
                tenant_id=tenant.id,
                target=kb_id,
                ip=client_ip(request),
                detail={"files": [r["filename"] for r in results], "accepted": accepted_docs},
            )
    except Exception:  # noqa: BLE001 - 审计失败不影响业务结果
        logger.exception("上传审计写入失败 kb=%s", kb_id)

    return {"items": results, "accepted": accepted_docs, "rejected": len(results) - accepted_docs}


@router.get("/documents/{doc_id}/assets")
def list_assets(
    doc_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    """列出文档里抽出的图片与识别状态。

    视觉模型可能认错数字（实测过把 MTK7621+7612 认成 MTK7621+761），
    识别结果会直接进知识库影响 AI 回答 —— 所以必须给人工核对入口，
    而不是把「模型说了什么」当成黑盒。
    """
    doc = scope.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    rows = scope.list(DocumentAsset, doc_id=doc_id)
    rows.sort(key=lambda a: a.seq)
    return {
        "items": [
            {
                "id": a.id,
                "seq": a.seq,
                "status": a.status,
                "mime": a.mime,
                "size_bytes": a.size_bytes,
                "width": a.width,
                "height": a.height,
                "page_index": a.page_index,
                "source": a.source,
                "description": a.description,
                "error": a.error,
                "image_url": f"/api/assets/{a.id}/image" if a.rel_path else "",
            }
            for a in rows
        ]
    }


@router.get("/assets/{asset_id}/image")
def get_asset_image(
    asset_id: str,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
):
    """返回图片原文。走 docstore 的路径校验，防止越界读任意文件。"""
    from fastapi.responses import FileResponse

    asset = scope.get(DocumentAsset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片资产不存在")
    path = docstore.abs_path(asset.rel_path)
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片文件不存在（可能在上传识别功能上线前入库，原图未落盘）",
        )
    return FileResponse(path, media_type=asset.mime or "image/png")


@router.patch("/documents/{doc_id}")
def toggle_doc(
    doc_id: str,
    enabled: bool,
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    doc = scope.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    doc.enabled = enabled
    for chunk in scope.db.execute(
        select(Chunk).where(Chunk.tenant_id == doc.tenant_id, Chunk.doc_id == doc_id)
    ).scalars().all():
        chunk.enabled = enabled
    scope.db.commit()
    return {"ok": True, "enabled": enabled}


@router.delete("/documents/{doc_id}")
def delete_doc(
    doc_id: str,
    request: Request,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    doc = scope.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    # 向量库清理是「可重建的投影」，失败只记告警，绝不能挡住删除主路径
    # （SQLite 才是真源）。这里必须兜住 BaseException：运行环境的安全删除
    # 守卫会 raise SystemExit，而它在 commit 之前冒泡 = 事务回滚 = 删不掉。
    try:
        purge_document(scope.db, tenant.id, doc_id)
    except BaseException as exc:  # noqa: BLE001 - 见 app/docstore.py remove_doc 的说明
        logger.warning("文档向量清理失败（已跳过，SQLite 侧照常删除）doc=%s：%s", doc_id, exc)
    kb = scope.get(KnowledgeBase, doc.kb_id)
    if kb is not None:
        kb.total_bytes = max(0, kb.total_bytes - doc.size_bytes)
    # 资产行没有对 documents 的外键约束（按 doc_id 弱关联），必须显式清理，
    # 否则文档删了、图片记录还在，列表里出现「幽灵图片」（与 delete_kb 同理）
    for asset in scope.list(DocumentAsset, doc_id=doc_id):
        scope.db.delete(asset)
    # 原文件与抽出的图片一起清掉 —— 留着只会占磁盘，且没有正文对应的图是死数据
    docstore.remove_doc(tenant.id, doc_id)
    scope.db.delete(doc)
    scope.db.commit()
    return {"ok": True}


@router.post("/documents/{doc_id}/reindex")
async def reindex_doc(
    doc_id: str,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    """重跑入库（换向量模型或切分参数后使用）。"""
    doc = scope.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    texts = [
        c.text
        for c in scope.db.execute(
            select(Chunk)
            .where(Chunk.tenant_id == tenant.id, Chunk.doc_id == doc_id)
            .order_by(Chunk.seq)
        ).scalars().all()
    ]
    if not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该文档没有已入库内容，请删除后重新上传")
    purge_document(scope.db, tenant.id, doc_id)
    # 把已有片段当原子块回灌：表格类文档每块就是一条记录，
    # 若退回 chunk_text 会被并段重新粘回一起，等于白切。
    res = await rag.index_document(
        scope.db,
        tenant_id=tenant.id,
        kb_id=doc.kb_id,
        doc_id=doc_id,
        text="\n\n".join(texts),
        blocks=texts,
    )
    doc.chunk_count = res.chunk_count
    # 重索引不重解析：解析期的提示（如某页未能提取）仍然有效，保留并重新分级，
    # 不能因为重跑了一遍向量化就把「内容有缺口」的事实抹成 success。
    notes = [s for s in (doc.error or "").split("；") if s] + list(res.warnings)
    doc.status = DocStatus.PARTIAL if _degrades_doc_status(notes) else DocStatus.SUCCESS
    if notes:
        doc.error = "；".join(notes)[:2000]
    scope.db.commit()
    return {"ok": True, "chunks": res.chunk_count}


@router.post("/knowledge/search-test")
async def search_test(
    payload: SearchTestIn,
    tenant: Tenant = Depends(get_tenant),
    scope: TenantScope = Depends(get_scope),
    user: User = Depends(editor),
) -> dict:
    """检索效果验证：只在本租户（或指定 AI 员工绑定的库）范围内检索。"""
    if payload.employee_id:
        employee = scope.get(AiEmployee, payload.employee_id)
        if employee is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI 员工不存在")
        kb_ids = rag.kb_ids_for_employee(scope.db, tenant.id, payload.employee_id)
    elif payload.kb_id:
        kb = scope.get(KnowledgeBase, payload.kb_id)
        if kb is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
        kb_ids = [payload.kb_id]
    else:
        kb_ids = [k.id for k in scope.list(KnowledgeBase)]

    hits, tokens = await rag.retrieve(
        scope.db,
        tenant_id=tenant.id,
        kb_ids=kb_ids,
        query=payload.query,
        top_k=payload.top_k,
        min_score=0.0,
    )
    return {
        "query": payload.query,
        "tokens": tokens,
        "items": [
            {
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "kb_id": h.kb_id,
                "filename": h.filename,
                "score": round(h.score, 4),
                "text": h.text,
            }
            for h in hits
        ],
    }
