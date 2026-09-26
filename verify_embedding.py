"""向量模型（检索召回）真机验证：确认 bge-m3 真的可用、维度对、语义有效。

与 smoke_test.py 的区别：
- smoke_test.py 打 HTTP 走端到端；
- 本脚本直接压 embedding 层，回答三个问题：
  1. **通不通**：EMBED_API_KEY / EMBED_MODEL / EMBED_BASE_URL 是否正确；
  2. **维度对不对**：模型实际输出维度 == EMBED_DIM（不等就必须重建向量库集合）；
  3. **语义有没有用**：相关文本的相似度必须**明显高于**无关文本。
     这一条最关键 —— 用本地占位向量时检索「看着能跑但全是噪音」，
     只有拿真实模型算相似度才能证明检索召回是有效的。

用法：
    python verify_embedding.py

前置：.env 里 EMBED_PROVIDER=api 且 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL 已填。
"""
from __future__ import annotations

import asyncio
import sys

from app.config import settings
from app.embedding import EmbeddingUnavailable, embed_texts
from app.vectorstore.base import cosine

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  <- {detail}" if detail else ""))


def banner(text: str) -> None:
    print(f"\n{text}")
    print("-" * 68)


# --------------------------------------------------------------------------- #
# 样本设计要点：**必须按「切分后的块」来测，不能按单句来测**。
# 单句对单句的相似度（0.69~0.71）远高于「短查询 vs 真实块」的相似度
# （真实块里混着别的主题，实测只有 0.53 左右）。用单句定出来的门槛会偏高，
# 上线后表现为「知识库里明明有，就是召不回来」。
# 所以这里用真实的 chunk_text 产出块，再拿短查询去打，测的就是生产真实形态。
# --------------------------------------------------------------------------- #
FAQ_DOC = "\n\n".join(
    [
        "配送与时效：现货商品在付款后 48 小时内发出，使用顺丰陆运，华东地区一般 2 天送达，西北地区 3-5 天。",
        "户外电源保修：户外电源整机保修 24 个月，电池组保修 12 个月。人为损坏、进液、私自拆机不在保修范围内。",
        "发票：支持开具电子普票，下单时在备注填写抬头与税号即可；专票需提供完整开票资料，开出后 3 个工作日内发送到邮箱。",
        "退换货：收到货 7 天内不影响二次销售可无理由退货，运费由买家承担；质量问题由我们承担来回运费。",
        "帐篷防水等级：本店三人帐篷采用 3000mm 防水涂层，可应对中雨；暴雨天气建议加装天幕。",
        "床垫试睡：床垫支持 100 天试睡，试睡期内不满意可申请退货，需保留原包装并承担退回运费。",
        "沙发送装：沙发默认含送装一体服务，下单后 3-5 个工作日预约上门，偏远地区需额外加收运费。",
        "内部信息：老客户专属折扣码 STAR20，仅限电话下单使用。",
    ]
)
# (查询, 答案所在段落的正文片段)
# 用**正文片段**而不是主题词来判定「这块是不是答案」：块之间有重叠，
# 主题词恰好落在切开处时（实测某块从"具电子普票"开始，"发票"两字被拆开了）
# 按关键词判会把正确答案误标成负样本，进而得出「模型区分度不足」的假结论。
TOPIC_QUERIES: list[tuple[str, str]] = [
    ("你们的退货政策是怎样的", "不影响二次销售可无理由退货"),
    ("老客户折扣码 STAR20", "折扣码 STAR20，仅限电话下单"),
    ("开发票需要什么资料", "在备注填写抬头与税号"),
]
OFF_TOPIC_QUERIES = [
    "今天北京天气怎么样",
    "公司年会什么时候办",
]


async def main() -> int:
    banner("0. 当前配置")
    print(f"  EMBED_PROVIDER = {settings.embed_provider}")
    print(f"  EMBED_BASE_URL = {settings.embed_base_url}")
    print(f"  EMBED_MODEL    = {settings.embed_model}")
    print(f"  EMBED_DIM      = {settings.embed_dim}  (期望等于模型真实输出维度)")
    print(f"  EMBED_BATCH    = {settings.embed_batch}")
    print(f"  MIN_SCORE      = {settings.min_score}")

    if settings.embed_provider != "api":
        print(
            "\n  !! 当前 EMBED_PROVIDER 不是 api，用的是本地占位向量（无真实语义能力）。\n"
            "     本脚本验证真实模型，请先在 .env 设 EMBED_PROVIDER=api。"
        )
        return 2

    banner("1. 连通性与鉴权")
    try:
        res = await embed_texts(["连通性检查"])
    except EmbeddingUnavailable as exc:
        print(f"  !! 调用失败：{exc.message}")
        if exc.detail:
            print(f"     detail: {exc.detail}")
        print("     排查顺序：")
        print("       1) EMBED_BASE_URL 是否为 https://ai.gitee.com/api/v1（要带 /api/v1）")
        print("       2) EMBED_API_KEY 是否有效（Gitee AI「设置 → 访问令牌」）")
        print("       3) EMBED_MODEL 是否真实存在（bge-m3）")
        print("       4) 本机出口网络 / 代理是否放行 ai.gitee.com")
        return 3

    check("接口可达且鉴权通过", True)
    check(
        "返回维度与 EMBED_DIM 一致",
        res.dim == settings.embed_dim,
        f"模型返回 {res.dim}，配置 {settings.embed_dim}",
    )
    print(f"  本次探测消耗 tokens = {res.tokens}（usage 字段可正常解析）")
    if res.dim != settings.embed_dim:
        print(
            f"     修法：把 EMBED_DIM 改成 {res.dim}；"
            f"若向量库集合已按 {settings.embed_dim} 维建过，"
            f"必须删掉集合并重新上传文档（不同维度/模型的向量不可混用）。"
        )
        return 3

    banner("2. 批量与顺序")
    texts = ["第一句", "第二句", "第三句"]
    batch = await embed_texts(texts)
    check("批量返回条数正确", len(batch.vectors) == 3, f"实际 {len(batch.vectors)}")
    singles = [await embed_texts([t]) for t in texts]
    same = all(
        cosine(batch.vectors[i], singles[i].vectors[0]) > 0.999 for i in range(len(texts))
    )
    check("批量结果与单条调用一致（顺序未错位）", same,
          "批量与单条的向量不一致，说明返回顺序或分批逻辑有问题")

    blank = await embed_texts(["", "   "])
    check("空文本/纯空白不报错且有向量", all(len(v) == res.dim for v in blank.vectors))

    banner("3. 语义有效性：短查询 vs 真实切分块（生产真实形态）")
    from app.rag import chunk_text

    chunks = chunk_text(FAQ_DOC)
    print(f"  演示文档 {len(FAQ_DOC)} 字，切分出 {len(chunks)} 个块：")
    for i, c in enumerate(chunks):
        print(f"    chunk[{i}] {len(c):>3} 字  {c[:34]}...")

    chunk_vecs = (await embed_texts(chunks)).vectors

    # MIN_SCORE 的职责只有两条，判据就照这两条来定，不要拿「所有正样本都高于
    # 所有负样本」这种理想条件去要求：块里本来就会混着相邻主题的句子
    # （比如"床垫试睡…可申请退货"对退货查询就是半相关），这类样本注定重叠。
    #   1) 正确答案所在块必须过线（否则漏召 = 明明有答案却答不上来）；
    #   2) 纯离题查询不能过线（否则噪音进提示词 = 答出像模像样的错话）。
    top1_positives: list[float] = []   # 各查询的 Top-1 命中分（应全部过线）
    noise_ceilings: list[float] = []   # 离题查询的最高分（应全部不过线）

    for query, snippet in TOPIC_QUERIES:
        qv = (await embed_texts([query])).vectors[0]
        ranked = sorted(
            ((cosine(qv, v), c) for v, c in zip(chunk_vecs, chunks)), key=lambda x: -x[0]
        )
        answer_hits = [(s, c) for s, c in ranked if snippet in c]
        best_score, best_chunk = ranked[0]
        print(f"\n  查询：{query}")
        for s, c in ranked[:3]:
            mark = "答案块" if snippet in c else "  其他"
            print(f"    {s:.4f} [{mark}] {c[:38]}...")
        check(f"Top-1 命中答案块（{query[:10]}）", snippet in best_chunk,
              f"Top-1 是 {best_chunk[:30]}... 得分 {best_score:.4f}")
        if answer_hits:
            top1_positives.append(max(s for s, _ in answer_hits))

    # 噪音只取**真正离题**的查询。同一文档里其他块不算噪音：
    # 它们常常半相关（"床垫试睡…可申请退货"对退货查询就是半相关），
    # 把它们当噪音会把门槛越推越高，最后变成「什么都不敢召回」。
    for query in OFF_TOPIC_QUERIES:
        qv = (await embed_texts([query])).vectors[0]
        top = max(cosine(qv, v) for v in chunk_vecs)
        print(f"\n  离题查询：{query} -> 最高 {top:.4f}（应当低于门槛）")
        noise_ceilings.append(top)

    lo, hi = max(noise_ceilings), min(top1_positives)
    print(f"\n  答案块得分下限 {hi:.4f}   离题查询得分上限 {lo:.4f}")

    # 门槛取 40% 位置（偏保守）：漏召只是降级转人工（安全失败），
    # 误召会让模型拿着不相干的段落编出听起来很像真的答案（最糟的失败）。
    suggested = round(lo + (hi - lo) * 0.4, 2) if lo < hi else round(lo * 0.9, 2)
    print(f"  建议 MIN_SCORE ≈ {suggested}（当前 {settings.min_score}）")
    print(f"     依据：噪音上限={lo:.4f}，答案下限={hi:.4f}，取中间偏保守的 40% 位置")
    check(
        "答案块能过门槛（不会漏召）",
        settings.min_score <= hi,
        f"门槛 {settings.min_score} 高于答案块得分下限 {hi:.4f}，会导致明明有答案却答不上来",
    )
    check(
        "离题查询过不了门槛（不会误召）",
        settings.min_score > lo,
        f"门槛 {settings.min_score} 不高于噪音上限 {lo:.4f}，无关段落会混进提示词",
    )

    banner("5. 错误分流：配错了要炸得明白")
    from app import embedding as emb

    saved = settings.embed_model
    try:
        settings.embed_model = "definitely-no-such-model-xyz"
        emb._BREAKERS.clear()
        try:
            await embed_texts(["测试"])
            check("模型名错误时应报错", False, "竟然成功了")
        except EmbeddingUnavailable as exc:
            detail = exc.detail or ""
            check("模型名错误时抛出明确的 EmbeddingUnavailable 并透出上游原文",
                  bool(detail), "detail 为空，排查时看不到上游报什么")
            print(f"     上游原文：{detail[:160]}")
            trips = emb._BREAKERS.get(
                (settings.embed_base_url.rstrip("/"), "definitely-no-such-model-xyz")
            )
            check("配置类错误不应触发熔断（否则会被误判成服务不稳定）",
                  trips is None or not trips.is_open())
    finally:
        settings.embed_model = saved
        emb._BREAKERS.clear()

    banner("结果")
    print(f"  通过 {PASS} 项，失败 {FAIL} 项")
    if FAIL:
        print("  提示：第 1/3 项失败通常意味着模型端点或 Key 有问题；")
        print("        第 3 项失败说明这个模型在你的语料上区分度不足，不适合做检索召回。")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
