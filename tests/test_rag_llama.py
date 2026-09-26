"""llama 引擎（RAG_ENGINE=llama）标准化组件的离线单测。

不联网、不碰 Qdrant：只验证
1. 节点构建：chunk hex id ↔ UUID 双向转换、metadata 落位
2. JiebaBM25Retriever：中文 jieba 预分词后 BM25 能召回词面相关块
3. RRF 融合：双路结果按倒数排名合并、按 node id 去重
4. 分流开关：默认 legacy，llama 时入口转发到 rag_llama
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import TextNode

from app import rag, rag_llama
from app.config import settings


# --------------------------------------------------------------------------- #
# 节点构建
# --------------------------------------------------------------------------- #
def test_node_id_roundtrip():
    chunk_id = "a" * 32  # 32 位 hex（new_id 的形态）
    node_id = rag_llama._node_id(chunk_id)
    # 必须是合法 UUID 字符串（Qdrant point id 约束）
    assert str(uuid.UUID(node_id)) == node_id
    assert rag_llama._chunk_id_from_node(node_id) == chunk_id


def test_make_node_metadata_and_ref():
    node = rag_llama._make_node(
        "b" * 32,
        "正文内容",
        tenant_id="t1",
        kb_id="kb1",
        doc_id="doc1",
        seq=3,
        vector=[0.1, 0.2],
    )
    assert node.node_id == rag_llama._node_id("b" * 32)
    assert node.embedding == [0.1, 0.2]
    assert node.ref_doc_id == "doc1"
    assert node.metadata["tenant_id"] == "t1"
    assert node.metadata["kb_id"] == "kb1"
    assert node.metadata["doc_id"] == "doc1"
    assert node.metadata["seq"] == 3


# --------------------------------------------------------------------------- #
# BM25 中文检索（jieba 预分词）
# --------------------------------------------------------------------------- #
def _bm25_nodes(texts: list[str]):
    import jieba

    nodes = []
    for i, t in enumerate(texts):
        nodes.append(
            TextNode(
                id_=rag_llama._node_id(f"{i:032x}"),
                text=" ".join(jieba.lcut(t)),
            )
        )
    return nodes


def test_jieba_bm25_recalls_lexical_match():
    from llama_index.core.schema import QueryBundle

    texts = [
        "本产品保修期为一年，保修范围内免费维修。",
        "首次使用前请充满电并阅读安全须知。",
        "清洁水箱时请勿使用腐蚀性洗涤剂。",
    ]
    retriever = rag_llama.JiebaBM25Retriever(
        nodes=_bm25_nodes(texts),
        similarity_top_k=2,
        token_pattern=r"\w+",
        skip_stemming=True,
        language="en",
    )
    hits = retriever.retrieve(QueryBundle(query_str="保修期是多长时间"))
    assert hits, "BM25 必须能召回词面相关的块"
    top_id = rag_llama._chunk_id_from_node(hits[0].node.node_id)
    assert top_id == "0" * 32, "保修提问的第一名应是保修块，实际返回了别的块"


# --------------------------------------------------------------------------- #
# 分流开关
# --------------------------------------------------------------------------- #
def test_engine_dispatch_default_is_legacy():
    """引擎字段的**代码默认值**必须是 legacy（.env 可切 llama，不影响默认安全值）。"""
    from app.config import Settings

    assert Settings.model_fields["rag_engine"].default == "legacy"


def test_engine_dispatch_routes_to_llama(monkeypatch):
    """RAG_ENGINE=llama 时 retrieve 应转发到 rag_llama.retrieve。"""
    called = {}

    async def fake_retrieve(*args, **kwargs):
        called["kwargs"] = kwargs
        return [], 0

    monkeypatch.setattr(settings, "rag_engine", "llama")
    monkeypatch.setattr(settings, "vector_backend", "qdrant")
    monkeypatch.setattr(rag_llama, "retrieve", fake_retrieve)

    class _DB:
        pass

    asyncio.run(
        rag.retrieve(
            _DB(),
            tenant_id="t1",
            kb_ids=["kb1"],
            query="测试",
        )
    )
    assert called["kwargs"]["tenant_id"] == "t1"
    monkeypatch.undo()


# --------------------------------------------------------------------------- #
# RRF 融合去重（纯函数级验证：同 id 双路命中只留一条）
# --------------------------------------------------------------------------- #
def test_fusion_dedupes_by_node_id():
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    class _StaticRetriever(BaseRetriever):
        def __init__(self, scores: dict[str, float]):
            super().__init__()
            self._scores = scores

        def _retrieve(self, query_bundle):
            return [
                NodeWithScore(node=TextNode(id_=nid, text=nid), score=sc)
                for nid, sc in self._scores.items()
            ]

    fusion = QueryFusionRetriever(
        [
            _StaticRetriever({"n1": 0.9, "n2": 0.5}),
            _StaticRetriever({"n2": 0.8, "n3": 0.4}),
        ],
        similarity_top_k=5,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,  # 0.14 默认 SIMPLE，显式锁定 RRF
        use_async=False,
    )
    hits = fusion.retrieve("查询")
    ids = [h.node.node_id for h in hits]
    assert len(ids) == len(set(ids)), "RRF 融合必须按 node id 去重"
    assert set(ids) == {"n1", "n2", "n3"}
    # 双路都命中的 n2 应排最前（RRF：两边都贡献排名分）
    assert ids[0] == "n2"
