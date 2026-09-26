#!/usr/bin/env python
"""检索层 A/B 评测：在同一个知识库 + 同一套题目上，横向对比 embedding 模型与 rerank 的召回质量。

为什么需要这个脚本
------------------
换 embedding / 加 rerank 之前，凭模型名或 MTEB 榜单猜测效果是不可靠的：
榜单用的是公开通用语料，而真实场景是「短口语 query vs 自家切分出来的块」。
必须用自家数据测：Recall@K（前 K 个块里能凑齐答案关键词的题目比例）。

判据
----
题目自带 check.groups（关键词组，组内是同义写法）。
Recall@K = 前 K 块合并文本命中**全部**关键词组的题目占比。
这是**严格**指标——宁可低估，也不要把「捞到一半」算成命中。

用法
----
    python verify_retrieval.py                         # 跑默认候选集
    python verify_retrieval.py --models bge-m3,Qwen3-Embedding-8B
    python verify_retrieval.py --rerank Qwen3-Reranker-8B
    python verify_retrieval.py --no-cache              # 忽略缓存重新烧 API

结果缓存在 eval/amazon_supply_chain/retrieval_cache.json，换题目或换文档才需 --no-cache。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "eval" / "amazon_supply_chain" / "retrieval_cache.json"

DEFAULT_EMBED_MODELS = ["bge-m3", "Qwen3-Embedding-8B", "Qwen3-Embedding-4B", "Qwen3-Embedding-0.6B"]
DEFAULT_RERANK_MODELS = ["Qwen3-Reranker-8B", "bge-reranker-v2-m3"]

# 粗排深度：rerank 的输入候选数。线上 TOP_K=5 太小（答案常在第 6 名开外），
# 所以粗排要放宽，再交给 rerank 精排回 5 条。
RERANK_CANDIDATES = 20


# --------------------------------------------------------------------------- #
# 数据装载
# --------------------------------------------------------------------------- #
def load_corpus() -> tuple[list[str], list[dict]]:
    """取 Amazon 手册的分块 + 题库。"""
    db = sqlite3.connect(ROOT / "aics.db")
    row = db.execute(
        "select id from documents where filename like '%amazon-supplier%' order by chunk_count desc limit 1"
    ).fetchone()
    if row is None:
        sys.exit("库里找不到 Amazon 供应链手册，请先上传该文档")
    chunks = [
        r[0]
        for r in db.execute(
            "select text from chunks where doc_id=? order by seq", (row[0],)
        )
    ]
    qpath = ROOT / "eval" / "amazon_supply_chain" / "questions.json"
    questions = json.load(open(qpath, encoding="utf-8"))["questions"]
    return chunks, questions


def is_gradable(q: dict) -> bool:
    """只有答案确实在文档里的题才参与召回评测。

    expect_fallback（正确答案=没查到）与 expect_canned（固定话术）考的不是召回，
    统计进去会把「检索该不该捞到」这件事搅浑，故排除。
    """
    return q["check"].get("type") in ("contains_all", "contains_any_n")


def hit_all(q: dict, text: str) -> bool:
    """按题目自己的判分规则判定：命中即该块集合能支撑正确答案。

    - contains_all：每组至少命中一个同义写法，且所有组都要命中；
    - contains_any_n：命中至少 n 组（n 由题目给出，缺省按全部）。
    """
    check = q["check"]
    groups = check.get("groups") or []
    if not groups:
        return False
    got = sum(1 for g in groups if any(w and w in text for w in g))
    if check.get("type") == "contains_any_n":
        return got >= int(check.get("n") or len(groups))
    return got == len(groups)


# --------------------------------------------------------------------------- #
# 上游调用（带缓存）
# --------------------------------------------------------------------------- #
class Upstream:
    def __init__(self, base: str, key: str, cache: dict, no_cache: bool):
        self.base = base.rstrip("/")
        self.key = key
        self.cache = cache
        self.no_cache = no_cache
        self.calls = 0

    def _hdr(self) -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        ck = f"emb::{model}::{len(texts)}"
        if not self.no_cache and ck in self.cache:
            return self.cache[ck]
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120) as cli:
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                r = await cli.post(
                    f"{self.base}/embeddings",
                    headers=self._hdr(),
                    json={"model": model, "input": batch},
                )
                self.calls += 1
                if r.status_code != 200:
                    sys.exit(f"embedding {model} 失败 {r.status_code}: {r.text[:200]}")
                items = sorted(r.json().get("data") or [], key=lambda x: x.get("index", 0))
                out.extend([list(map(float, it["embedding"])) for it in items])
        self.cache[ck] = out
        return out

    async def rerank(self, model: str, query: str, docs: list[str]) -> list[float]:
        # 缓存 key **必须带上候选集合**：rerank 分数是 query 与具体候选两两算出来的，
        # 只按 (模型, query) 缓存会让不同 embedding 粗排出的候选共用同一份分数，
        # 表现为「换 embedding 后加 rerank 反而变差」这种假结论（实测踩过）。
        ck = f"rr::{model}::{hashlib.md5((query + '\x00' + '\n'.join(docs)).encode()).hexdigest()}"
        if not self.no_cache and ck in self.cache:
            return self.cache[ck]
        async with httpx.AsyncClient(timeout=120) as cli:
            r = await cli.post(
                f"{self.base}/rerank",
                headers=self._hdr(),
                json={"model": model, "query": query, "documents": docs},
            )
        self.calls += 1
        if r.status_code != 200:
            sys.exit(f"rerank {model} 失败 {r.status_code}: {r.text[:200]}")
        res = sorted(r.json().get("results") or [], key=lambda x: x.get("index", 0))
        scores = [float(x.get("relevance_score", 0)) for x in res]
        self.cache[ck] = scores
        return scores


def cos(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
async def run(models: list[str], rerank_models: list[str], up: Upstream) -> None:
    chunks, questions = load_corpus()
    queries = [q["question"] for q in questions]
    gradable = sum(1 for q in questions if is_gradable(q))
    print(f"语料 {len(chunks)} 块 / 题目 {len(questions)} 道（其中 {gradable} 道可用于召回评测）\n")

    results: dict[str, dict] = {}
    for model in models:
        qv = await up.embed(model, queries)
        dv = await up.embed(model, chunks)
        # 逐题排序
        ranked: list[list[int]] = []
        for q in qv:
            order = sorted(range(len(dv)), key=lambda j: cos(q, dv[j]), reverse=True)
            ranked.append(order)
        rec = {k: recall_at_k(ranked, chunks, questions, k) for k in (1, 3, 5, 10)}
        results[model] = rec
        print(f"  {model:24s} " + "  ".join(f"R@{k}={rec[k]*100:5.1f}%" for k in (1, 3, 5, 10)))

        for rm in rerank_models:
            rec2 = await rerank_eval(rm, ranked, chunks, queries, questions, up)
            results[f"{model} + {rm}"] = rec2
            print(
                f"  {'└─ ' + rm:24s} "
                + "  ".join(f"R@{k}={rec2[k]*100:5.1f}%" for k in (1, 3, 5))
            )
        print()

    print("=== 汇总（R@5 为线上 TOP_K=5 的实际口径）===")
    for name, rec in sorted(results.items(), key=lambda x: -x[1][5]):
        print(f"  {rec[5]*100:5.1f}%  R@1={rec[1]*100:5.1f}%  R@3={rec[3]*100:5.1f}%   {name}")


def recall_at_k(ranked, chunks, questions, k: int) -> float:
    ok = total = 0
    for qi, order in enumerate(ranked):
        if not is_gradable(questions[qi]):
            continue
        total += 1
        text = "\n".join(chunks[j] for j in order[:k])
        if hit_all(questions[qi], text):
            ok += 1
    return ok / total if total else 0.0


async def rerank_eval(model, ranked, chunks, queries, questions, up) -> dict:
    """粗排取前 N 条 → rerank 精排 → 再算 Recall@K。"""
    new_ranked: list[list[int]] = []
    for qi, order in enumerate(ranked):
        cands = order[:RERANK_CANDIDATES]
        scores = await up.rerank(model, queries[qi], [chunks[j] for j in cands])
        new_ranked.append([j for _, j in sorted(zip(scores, cands), reverse=True)])
    return {k: recall_at_k(new_ranked, chunks, questions, k) for k in (1, 3, 5)}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_EMBED_MODELS))
    ap.add_argument("--rerank", default=",".join(DEFAULT_RERANK_MODELS))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    base = os.getenv("EMBED_BASE_URL", "")
    key = os.getenv("EMBED_API_KEY", "")
    if not base or not key:
        sys.exit("缺少 EMBED_BASE_URL / EMBED_API_KEY，请检查 .env")

    cache = {}
    if CACHE.exists() and not args.no_cache:
        cache = json.load(open(CACHE, encoding="utf-8"))
    up = Upstream(base, key, cache, args.no_cache)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rrs = [] if args.no_rerank else [m.strip() for m in args.rerank.split(",") if m.strip()]

    asyncio.run(run(models, rrs, up))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(up.cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n上游调用 {up.calls} 次，缓存已写入 {CACHE.name}")


if __name__ == "__main__":
    main()
