"""删除文档（DELETE /api/documents/{doc_id}）的回归测试。

覆盖本次修复的三条关键性质：
1. 三张表（documents / chunks / document_assets）都要真删干净，不能留「幽灵图片」；
2. Qdrant 不可达时降级为 warning 而非 500，且异常不能被完全吞掉（要有日志）；
3. document_assets 的清理必须带 tenant_id 过滤，不能删到别的租户的行。
"""
from __future__ import annotations

import logging
import uuid

import pytest
from sqlalchemy import select

from app.database import SessionLocal
from app.models import Chunk, Document, DocumentAsset, KnowledgeBase
from app.scoping import TenantScope

DOC_NAME = "qa_delete_regression.txt"
DOC_BODY = "删除回归测试文档：帐篷使用后应立即晾干，睡袋定期清洗。".encode()

# 本文件会往演示租户的默认知识库里塞文档。如果不清理，这些多余片段会挤占
# 检索 top_k，让 test_flows 的召回断言（如「户外电源保修」）失真 ——
# 所以所有自己造的文档/知识库都登记在这里，用例结束统一清掉。
_CREATED_DOCS: list[str] = []
_CREATED_KBS: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup_created(client, star):
    """用例结束后删掉本用例登记的文档与知识库，避免污染其它测试。"""
    yield
    for doc_id in reversed(_CREATED_DOCS):
        client.delete(f"/api/documents/{doc_id}", headers=star)
    _CREATED_DOCS.clear()
    for kb_id in reversed(_CREATED_KBS):
        client.delete(f"/api/knowledge-bases/{kb_id}", headers=star)
    _CREATED_KBS.clear()


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _first_kb(client, headers) -> dict:
    kbs = client.get("/api/knowledge-bases", headers=headers).json()
    items = kbs["items"] if isinstance(kbs, dict) else kbs
    return items[0]


def _upload(client, headers, kb_id, name=DOC_NAME, body=DOC_BODY) -> str:
    # 后端按「同名 + 同大小」判重，session 级 client 下同名同大小会被直接拒，
    # 所以每个用例用带 uuid 的独立文件名。
    name = f"{name.rsplit('.', 1)[0]}_{uuid.uuid4().hex[:8]}.txt"
    res = client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=headers,
        files=[("files", (name, body, "text/plain"))],
    )
    assert res.status_code == 200, res.text
    items = res.json()["items"]
    assert items, res.text
    # 入库失败时后端也会建 Document 行（status=failed），但响应里可能不带 doc_id；
    # 这种行同样要能被删除接口清掉，所以回查一次列表拿 id。
    doc_id = items[0].get("doc_id")
    if not doc_id:
        listing = client.get("/api/documents", headers=headers).json()["items"]
        match = [d for d in listing if d["filename"] == name]
        assert match, f"上传响应与列表都找不到 {name}：{res.text}"
        doc_id = match[0]["id"]
    _CREATED_DOCS.append(doc_id)
    return doc_id


def _seed_asset(db, tenant_id: str, kb_id: str, doc_id: str, seq: int = 1) -> str:
    """手工塞一条图片资产（离线测试不跑视觉识别，upload 不会自动产生 assets）。"""
    asset = DocumentAsset(
        id=f"qa-asset-{doc_id[:8]}-{seq}",
        tenant_id=tenant_id,
        kb_id=kb_id,
        doc_id=doc_id,
        seq=seq,
        source="qa",
        mime="image/png",
        size_bytes=10,
        width=1,
        height=1,
    )
    db.add(asset)
    db.commit()
    return asset.id


def _count(db, model, doc_id: str, tenant_id: str | None = None) -> int:
    """按 doc_id 统计。Document 表的主键就叫 id，其余表才有 doc_id 列。"""
    key = "id" if model is Document else "doc_id"
    stmt = select(model).where(getattr(model, key) == doc_id)
    if tenant_id is not None:
        stmt = stmt.where(model.tenant_id == tenant_id)
    return len(list(db.execute(stmt).scalars().all()))


@pytest.fixture
def qa_doc(client, star):
    """上传一个专属测试文档，返回 (doc_id, kb_id, tenant_id)。

    刻意自建隔离知识库：往演示默认库里塞文档会挤占检索 top_k，
    让 test_flows / test_retrieve_isolation 的召回断言失真。
    """
    created = client.post("/api/knowledge-bases", headers=star,
                          json={"name": "qa 删除回归隔离库", "description": "qa"})
    assert created.status_code in (200, 201), created.text
    kb_id = created.json()["id"]
    _CREATED_KBS.append(kb_id)
    doc_id = _upload(client, star, kb_id)
    tenant_id = client.get("/api/auth/me", headers=star).json()["tenant"]["id"]
    yield doc_id, kb_id, tenant_id


@pytest.fixture
def warn_logs():
    """捕获 aics.* 的 WARNING 及以上日志。

    本项目用 logging.basicConfig 接管了根 logger，pytest 自带的 caplog 在这套
    环境里不可用，所以自己挂一个 handler，语义等价且不依赖插件。
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    prev = root.level
    root.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(prev)


def _warnings_containing(records, needle: str) -> int:
    return sum(needle in r.getMessage() for r in records)


# --------------------------------------------------------------------------- #
# 1. 主路径：删除后三表无残留
# --------------------------------------------------------------------------- #
def test_delete_removes_rows_from_all_three_tables(client, star, qa_doc):
    doc_id, kb_id, tenant_id = qa_doc

    db = SessionLocal()
    try:
        _seed_asset(db, tenant_id, kb_id, doc_id)
        assert _count(db, Document, doc_id) == 1
        assert _count(db, DocumentAsset, doc_id) >= 1
    finally:
        db.close()

    res = client.delete(f"/api/documents/{doc_id}", headers=star)
    assert res.status_code == 200, res.text
    assert res.json() == {"ok": True}

    db = SessionLocal()
    try:
        # 三张表都要干净 —— 少了 document_assets 就会留下「幽灵图片」
        assert _count(db, Document, doc_id) == 0, "documents 表有残留"
        assert _count(db, Document, doc_id, tenant_id) == 0, "documents 表有残留"
        assert _count(db, Chunk, doc_id) == 0, "chunks 表有残留"
        assert _count(db, DocumentAsset, doc_id) == 0, "document_assets 表有残留"
    finally:
        db.close()


def test_deleted_doc_disappears_from_listing(client, star, qa_doc):
    doc_id, _kb, _t = qa_doc
    assert any(d["id"] == doc_id for d in client.get("/api/documents", headers=star).json()["items"])
    client.delete(f"/api/documents/{doc_id}", headers=star)
    assert not any(d["id"] == doc_id for d in client.get("/api/documents", headers=star).json()["items"])


# --------------------------------------------------------------------------- #
# 2. 幂等性
# --------------------------------------------------------------------------- #
def test_repeated_delete_returns_404_not_500(client, star, qa_doc):
    doc_id, _kb, _t = qa_doc
    assert client.delete(f"/api/documents/{doc_id}", headers=star).status_code == 200
    # 关键回归点：旧实现这里会 500（Qdrant 抖动 + 找不到文档）
    assert client.delete(f"/api/documents/{doc_id}", headers=star).status_code == 404
    assert client.delete(f"/api/documents/{doc_id}", headers=star).status_code == 404


def test_delete_unknown_doc_returns_404(client, star):
    assert client.delete("/api/documents/does-not-exist-id", headers=star).status_code == 404


def test_delete_requires_auth(client, qa_doc):
    doc_id, _kb, _t = qa_doc
    assert client.delete(f"/api/documents/{doc_id}").status_code in (401, 403)


# --------------------------------------------------------------------------- #
# 3. 租户隔离
# --------------------------------------------------------------------------- #
def test_cannot_delete_other_tenant_doc(client, star, sea):
    """star 拿不到 sea 的文档，也就删不掉 —— scope.get 过滤 tenant_id。"""
    kb = _first_kb(client, sea)
    doc_id = _upload(client, sea, kb["id"], name="qa_sea_owned.txt")

    res = client.delete(f"/api/documents/{doc_id}", headers=star)
    assert res.status_code == 404, res.text

    db = SessionLocal()
    try:
        assert _count(db, Document, doc_id) == 1, "sea 的文档被 star 删掉了！"
    finally:
        db.close()

    client.delete(f"/api/documents/{doc_id}", headers=sea)  # 清理


def test_asset_cleanup_is_tenant_scoped(client, star, sea, qa_doc):
    """资产清理必须带 tenant_id 过滤，不能误删同 doc_id 的其他租户行。

    这里用「故意构造一条别租户的行」来证明过滤真的生效：
    假如 delete_doc 里写成 `select(DocumentAsset).where(doc_id == doc_id)`，
    这条海蓝的资产就会跟着被删掉。
    """
    doc_id, kb_id, tenant_id = qa_doc
    sea_tenant = client.get("/api/auth/me", headers=sea).json()["tenant"]["id"]

    db = SessionLocal()
    try:
        _seed_asset(db, tenant_id, kb_id, doc_id, seq=1)
        alien = _seed_asset(db, sea_tenant, kb_id, doc_id, seq=99)
        assert _count(db, DocumentAsset, doc_id) == 2
    finally:
        db.close()

    assert client.delete(f"/api/documents/{doc_id}", headers=star).status_code == 200

    db = SessionLocal()
    try:
        # 本租户的资产被清掉
        assert _count(db, DocumentAsset, doc_id, tenant_id) == 0, "本租户资产未清理"
        # 关键断言：别租户的资产必须还在
        assert db.get(DocumentAsset, alien) is not None, "误删了其他租户的 document_assets 行！"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 4. Qdrant 不可达时的降级
# --------------------------------------------------------------------------- #
def test_delete_succeeds_when_vector_store_down(client, star, qa_doc, monkeypatch, warn_logs):
    """向量库不可达时删除仍要成功（200），且必须留下 warning 日志。

    异常不能被完全吞掉 —— 否则线上出问题就变成无声无息的数据不一致。

    注意降级点是「向量清理」而不是整个 purge_document：
    路由里并没有兜底 try/except，兜底在 rag.purge_document / rag_llama._purge_llama_points
    内部。这里按真实实现来测 —— 打断向量库，验证 SQLite 侧照常删干净。
    """
    doc_id, kb_id, tenant_id = qa_doc

    class _BoomStore:
        def delete_by_doc(self, *_a, **_k):
            raise ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")

        def delete_by_kb(self, *_a, **_k):
            raise ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")

    monkeypatch.setattr("app.rag.build_vector_store", lambda _db: _BoomStore())

    res = client.delete(f"/api/documents/{doc_id}", headers=star)
    assert res.status_code == 200, f"向量库不可达导致删除 500：{res.status_code} {res.text}"

    db = SessionLocal()
    try:
        assert _count(db, Document, doc_id) == 0, "SQLite 侧没有照常删除"
        assert _count(db, Chunk, doc_id) == 0, "chunks 没有照常清理"
    finally:
        db.close()

    assert _warnings_containing(warn_logs, "向量清理失败") >= 1, (
        "降级了但没有 warning 日志 —— 异常被完全吞掉，线上无法排查"
    )


def test_llama_purge_swallows_qdrant_error_with_warning(monkeypatch, warn_logs):
    """_purge_llama_points 自身：Qdrant 构造/探测/删除任一环节炸了都不能抛。

    这条覆盖线上真实故障点 —— 报 500 的根因就在 llama_vector_store() 构造阶段。
    """
    from app import rag_llama

    class _Boom:
        def __getattr__(self, _name):
            def _raise(*_a, **_k):
                raise ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")
            return _raise

    monkeypatch.setattr(rag_llama, "llama_vector_store", lambda: _Boom())

    # 不抛异常即通过（修复前这里会抛 ConnectionError 一路冒到路由层变成 500）
    rag_llama._purge_llama_points("tenant-x", {"doc_id": "doc-y"})

    assert _warnings_containing(warn_logs, "向量清理失败") >= 1, (
        "降级了但没有 warning 日志 —— 异常被完全吞掉，线上无法排查"
    )


def test_llama_purge_kb_swallows_qdrant_error(monkeypatch, warn_logs):
    """知识库级删除走同一个 _purge_llama_points，也必须降级。"""
    from app import rag_llama

    class _Boom:
        def __getattr__(self, _name):
            def _raise(*_a, **_k):
                raise ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")
            return _raise

    monkeypatch.setattr(rag_llama, "llama_vector_store", lambda: _Boom())
    rag_llama._purge_llama_points("tenant-x", {"kb_id": "kb-y"})
    assert _warnings_containing(warn_logs, "向量清理失败") >= 1


def test_legacy_purge_document_degrades_to_warning(monkeypatch, warn_logs, client, star):
    """legacy（db 向量后端）路径的 purge_document/purge_kb 也要降级。"""
    from app import rag as rag_mod

    monkeypatch.setattr(rag_mod, "_use_llama_engine", lambda: False)

    class _BoomStore:
        def delete_by_doc(self, *_a, **_k):
            raise ConnectionError("legacy store down")

        def delete_by_kb(self, *_a, **_k):
            raise ConnectionError("legacy store down")

    monkeypatch.setattr(rag_mod, "build_vector_store", lambda _db: _BoomStore())

    # 必须自建一个隔离知识库来测：purge_kb 会把该kb 下所有 chunk 清空，
    # 如果拿演示租户的默认库来测，就会连带删掉 test_retrieve_isolation /
    # test_flows 依赖的 bootstrap 片段，属于测试自己污染环境。
    created = client.post("/api/knowledge-bases", headers=star,
                          json={"name": "qa legacy purge 隔离库", "description": "qa"})
    assert created.status_code in (200, 201), created.text
    kb_id = created.json()["id"]
    _CREATED_KBS.append(kb_id)
    tenant_id = client.get("/api/auth/me", headers=star).json()["tenant"]["id"]
    doc_id = _upload(client, star, kb_id)

    db = SessionLocal()
    try:
        db.add(Chunk(id="qa-legacy-chunk", tenant_id=tenant_id, kb_id=kb_id,
                     doc_id=doc_id, seq=999, text="legacy 降级测试片段"))
        db.commit()

        # 不抛异常，且仍要照常删掉 SQLite 侧的 chunk
        rag_mod.purge_document(db, tenant_id, doc_id)
        rag_mod.purge_kb(db, tenant_id, kb_id)
        db.commit()

        left = db.execute(
            select(Chunk).where(Chunk.kb_id == kb_id)
        ).scalars().all()
        assert left == [], "legacy 路径降级后 SQLite 侧 chunk 未清理"
    finally:
        db.close()

    assert _warnings_containing(warn_logs, "向量清理失败") >= 2, (
        "legacy 路径两条降级都应有 warning 日志"
    )


# --------------------------------------------------------------------------- #
# 5. 知识库级联删除的资产清理（对称性）
# --------------------------------------------------------------------------- #
def test_delete_kb_also_clears_assets(client, star):
    """delete_kb 同样不能留幽灵资产（与 delete_doc 对称）。

    演示租户的默认知识库被 AI 员工绑定着，删不掉，所以自建一个干净库来测。
    """
    created = client.post("/api/knowledge-bases", headers=star,
                          json={"name": "qa 删除级联库", "description": "qa"})
    assert created.status_code in (200, 201), created.text
    kb_id = created.json()["id"]
    _CREATED_KBS.append(kb_id)
    tenant_id = client.get("/api/auth/me", headers=star).json()["tenant"]["id"]
    doc_id = _upload(client, star, kb_id, name="qa_kb_asset.txt")

    db = SessionLocal()
    try:
        _seed_asset(db, tenant_id, kb_id, doc_id)
    finally:
        db.close()

    res = client.delete(f"/api/knowledge-bases/{kb_id}", headers=star)
    assert res.status_code == 200, res.text

    db = SessionLocal()
    try:
        assert _count(db, Document, doc_id) == 0
        assert _count(db, DocumentAsset, doc_id) == 0, "删除知识库后仍有幽灵资产"
    finally:
        db.close()