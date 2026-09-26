"""切分粒度的回归测试。

这些用例锁住的是「接了真实语义模型之后才暴露」的问题：
把多个无关主题并进同一个块，会让 bge-m3 对其中任何一个主题的**具体提问**
都匹配不上（实测：6 条 FAQ 并成一个 302 字块时，直问折扣码只有 0.51，
低于召回门槛；独立成块是 0.78）。所以并段必须有上限。
"""
from __future__ import annotations

from app.config import settings
from app.rag import chunk_text

# 8 条互不相关的客服 FAQ —— 演示租户知识库的真实形态（约 400 字）
FAQ = "\n\n".join(
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


def test_multi_topic_doc_is_not_merged_into_one_chunk():
    """多主题文档不能被并成一个块 —— 否则任何具体提问都匹配不上。"""
    chunks = chunk_text(FAQ)
    assert len(chunks) > 1, f"8 个主题只切出 {len(chunks)} 块，语义会被稀释"


def test_very_short_doc_may_stay_one_chunk():
    """如实记录边界：整篇比 min_fill 还短时合成一块是**预期行为**。

    并段本来就是为了凑够可读长度；一篇才一两百字的文档整个成块没有坏处
    （反而是切碎了更糟）。这条用例是为了防止有人把上面那条断言理解成
    「任何文档都必须多块」而误改切分逻辑。
    """
    short = "\n\n".join(["第一条：内容很短。", "第二条：内容也很短。"])
    assert len(chunk_text(short)) == 1


def test_each_topic_lands_in_a_small_enough_chunk():
    """每个块不应塞满 chunk_size，否则又回到「一块多主题」。"""
    chunks = chunk_text(FAQ)
    assert all(len(c) <= settings.chunk_size for c in chunks)
    # 平均粒度应明显小于上限：并段只为了解决碎块，不是为了填满
    avg = sum(len(c) for c in chunks) / len(chunks)
    assert avg <= settings.chunk_size * 0.75, f"平均块长 {avg:.0f} 偏大"


def test_short_paragraphs_are_still_merged():
    """并段的初衷是避免碎块：一堆十几个字的段落不该一段一块。"""
    tiny = "\n\n".join("短句%02d，内容很少。" % i for i in range(8))
    chunks = chunk_text(tiny)
    assert len(chunks) < 8, f"8 个短段被切成 {len(chunks)} 块，碎块太多"
    assert len(chunks) >= 1


def test_min_fill_zero_means_one_block_per_chunk():
    """min_fill=0 是显式逃生口：需要逐段切时可以关掉并段。"""
    chunks = chunk_text(FAQ, min_fill=0)
    assert len(chunks) == 8, f"关掉并段后应为 8 块，实际 {len(chunks)}"


def test_no_content_is_lost():
    chunks = chunk_text(FAQ)
    merged = "".join(chunks)
    for topic in FAQ.split("\n\n"):
        assert topic in merged, f"片段丢失：{topic[:16]}"


def test_long_single_paragraph_is_windowed():
    """超长单段按窗口切，且每块不超过 size。"""
    long_text = "这是一句很长的说明。" * 200  # 2000 字，无空行
    chunks = chunk_text(long_text, size=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_empty_text_returns_nothing():
    assert chunk_text("") == []
    assert chunk_text(None) == []
    assert chunk_text("   \n\n  ") == []
