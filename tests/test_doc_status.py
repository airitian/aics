"""文档状态分级：warnings 只有「影响内容完整性/可信度」才把文档压成 partial。

背景：以前 `partial if warnings else success` 是一票否决，
「剥离了 1 行页眉」这种无害提示也会让文档显示「部分成功」，用户以为出了问题。
"""
from app.routers.knowledge import _degrades_doc_status


# --------------------------------------------------------------------------- #
# 纯处理说明：内容无缺失，不应拉低状态
# --------------------------------------------------------------------------- #
def test_info_warnings_do_not_degrade():
    assert _degrades_doc_status([]) is False
    assert not _degrades_doc_status(["已剥离 2 个跨页重复的页眉/页脚行"])
    assert not _degrades_doc_status(["已跳过 806 个标识符片段的向量化（仅支持精确号码查询）"])
    assert not _degrades_doc_status(["12 张图片已识别为文字并入库"])
    assert not _degrades_doc_status(
        ["1 张图片来自已转写的扫描页，整页文字已入库，不再重复识别"]
    )
    assert not _degrades_doc_status(["3 张图片过小或重复，已作为装饰元素跳过"])
    assert not _degrades_doc_status(["检测到首列存在合并单元格，已向下填充"])
    assert not _degrades_doc_status(["首行疑似分组表头（非字段名），已按多行表头合并"])
    assert not _degrades_doc_status(["表格形态无法判定，已保守按行切分；可在文件详情切换为按列解析"])
    assert not _degrades_doc_status(["未能识别分隔符，已按逗号处理"])
    assert not _degrades_doc_status(["文档结构读取异常，已按纯段落处理：boom"])


# --------------------------------------------------------------------------- #
# 内容完整性/可信度受损：必须保持 partial
# --------------------------------------------------------------------------- #
def test_content_gap_warnings_degrade():
    assert _degrades_doc_status(["第 35 页未能提取到文字（可能是扫描图片页），内容未入库"])
    assert _degrades_doc_status(
        ["第 1、2 页为扫描/图片页，已通过视觉模型转写文字入库（识别可能有误差）"]
    )
    assert _degrades_doc_status(["扫描页超过单文档转写上限（30 页），第 31 页 未做识别"])
    assert _degrades_doc_status(["文档含 5 张图片，当前未启用图片识别，图中内容未能入库。"])
    assert _degrades_doc_status(["2 张图片未能识别，AI 遇到这部分内容会如实说读不到"])
    assert _degrades_doc_status(["图片外链无法下载，已跳过：https://x.example/a.png"])
    assert _degrades_doc_status(["工作表「附录」无有效数据行，已跳过"])


def test_mixed_warnings_degrade_conservatively():
    """info 与内容缺口混在一起：只要有一条缺口就 partial。"""
    assert _degrades_doc_status(
        [
            "已剥离 3 个跨页重复的页眉/页脚行",
            "8 张图片已识别为文字并入库",
            "第 12 页未能提取到文字（可能是扫描图片页），内容未入库",
        ]
    )


def test_unknown_warning_defaults_to_degrade():
    """白名单外的未知提示宁可多标不漏标（退回旧行为也只是多显示一个 partial）。"""
    assert _degrades_doc_status(["某种未来新增的未知异常"])
