"""意图识别与转人工判定。

本期用「确定性规则 + 检索置信度」实现，可解释、可测试、零额外调用成本。
后续要换成模型分类器时，只需替换 detect_* 与 confidence 的来源，
调用方（app/agent.py）的接口不变。

注意：高风险话题清单属于 PRD 待确认项 #4，这里的清单是**可配置的业务参数**，
不是最终版。改清单只改本文件顶部的常量。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 规则清单（可配置业务参数）
# --------------------------------------------------------------------------- #
HANDOFF_INTENT_PATTERNS = [
    "转人工", "人工客服", "找人工", "要人工", "接人工", "真人", "人工服务", "客服人工",
    "转接人工", "人工坐席", "转客服", "找客服",
]

# 面向海外客户：意图识别是**关键词规则**，只写中文的话，外国客户说
# "I want to speak to a human" 匹配不上 → handoff 不触发，但模型看到这句话
# 会礼貌地口头承诺"马上为您转接"，客户以为转了人工、实际没人接单。
# 这里补常见语种的等价说法（短语，不用单词宽匹配以免误伤正常咨询）。
HANDOFF_INTENT_PATTERNS += [
    # 英/法/德等拉丁语系
    "human agent", "human support", "real person", "speak to a human", "talk to a human",
    "transfer me", "live agent", "customer service agent", "get a human", "human operator",
    "attendant humain", "parler à un humain", "agente humano", "hablar con un humano",
    "atender humano", "menschen sprechen", "mitarbeiter verbinden",
    # 俄语 / 日语 / 韩语
    "оператор", "живой человек", "соедините с", "переведите на",
    "担当者につないで", "人間の担当", "オペレーター", "상담원 연결",
]

# 明确高风险：出现即强制转人工
FORCED_HANDOFF_PATTERNS = [
    "投诉", "举报", "曝光", "起诉", "律师", "消协", "12315", "工商局", "媒体",
    "赔偿", "纠纷", "仲裁", "欺诈", "骗子", "起诉你们",
    # 讨价还价类（价格谈判）
    "能不能便宜", "便宜点", "少点钱", "打个折", "优惠点", "最低多少", "给个底价", "议价",
]

# 高风险话题也必须有外语版本：外国客户说 "I'll sue you" 若识别不到，
# 就不会走强制转人工，机器人自己接着聊，风险极高。
FORCED_HANDOFF_PATTERNS += [
    "complaint", "complain about", "report you", "expose you", "sue", "lawyer", "attorney",
    "fraud", "scam", "rip off", "consumer protection", "arbitration", "compensation",
    "жалоба", "жаловаться", "суд", "адвокат", "мошенник", "обман",
    "queja", "demanda", "abogado", "estafa", "fraude", "escroquerie", "betrug", "anwalt",
    "詐欺", "訴える", "弁護士", "クレーム",
    # 议价（价格谈判）
    "give me a discount", "any discount", "can you do a discount", "cheaper price",
    "lower the price", "best price", "final price", "bottom price", "negotiate",
    "make it cheaper", "скидку", "дешевле", "descuento", "rabatt", "値下げ", "安くして",
]

# 退款/退货：本身是正常业务问题，只有叠加负面情绪才升级为强制转人工
RISK_HINT_PATTERNS = ["退款", "退货", "退钱", "换货", "差价",
                      "refund", "money back", "return the", "exchange the", "price difference"]

NEGATIVE_PATTERNS = [
    "不满意", "太差", "差评", "垃圾", "坑", "被骗", "离谱", "什么意思", "你们怎么回事",
    "必须给个说法", "太过分",
    "unacceptable", "terrible", "awful", "useless", "disappointed", "angry", "ridiculous",
    "what the hell", "are you kidding", "worst",
]

ABUSE_PATTERNS = ["傻逼", "滚", "白痴", "废物", "去死", "妈的", "操你",
                  "idiot", "stupid", "moron", "shut up", "fuck", "shit", "damn", "screw you"]

MEANINGLESS_PATTERNS = {"在?", "在？", "在吗", "在么", "1", "。", ".", "?", "？", "hi", "hello", "你好",
                        "有人吗", "test", "11", "hey", "hey?", "yo", "ok", "hi?"}

AD_PATTERNS = ["加微信", "加v", "加V", "代购", "刷单", "兼职", "引流", "互赞", "推广",
               "add my wechat", "add wechat", "promotion service", "dropship", "drop ship",
               "brush orders", "buy followers", "collaborate with us", "advertise here"]

_SPLIT_RE = re.compile(r"[。！？!?；;\n]+|(?:另外|还有|以及|然后|顺便)")


@dataclass
class IntentResult:
    handoff: bool = False
    reason: str = ""
    forced: bool = False
    abuse: bool = False
    ad: bool = False
    meaningless: bool = False
    sub_intents: list[str] = field(default_factory=list)
    language: str = ""

    @property
    def multi_intent(self) -> bool:
        return len(self.sub_intents) > 1


def _hit(text: str, patterns: list[str]) -> str:
    low = text.lower()
    for p in patterns:
        if p.lower() in low:
            return p
    return ""


def detect_language(text: str) -> str:
    """粗粒度语言判定，用于「自动匹配访客语言」。"""
    if not text:
        return ""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    kana = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff")
    hangul = sum(1 for ch in text if "\uac00" <= ch <= "\ud7af")
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    thai = sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    total = cjk + kana + hangul + cyr + thai + latin
    if total == 0:
        return ""
    if kana > 0:
        return "ja"
    if hangul > 0:
        return "ko"
    if thai > 0:
        return "th"
    if cyr / total > 0.4:
        return "ru"
    if cjk / total > 0.3:
        return "zh-CN"
    if latin / total > 0.5:
        return "en"
    return ""


def split_sub_intents(text: str, limit: int = 4) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) >= 2 and p not in out:
            out.append(p)
    if not out:
        out = [text.strip()]
    return out[:limit]


def is_meaningless(text: str) -> bool:
    stripped = (text or "").strip()
    if stripped.lower() in MEANINGLESS_PATTERNS:
        return True
    if len(stripped) <= 1:
        return True
    return False


def detect(text: str) -> IntentResult:
    raw = text or ""
    result = IntentResult(sub_intents=split_sub_intents(raw))
    result.language = detect_language(raw)

    if _hit(raw, ABUSE_PATTERNS):
        result.abuse = True
    if _hit(raw, AD_PATTERNS):
        result.ad = True
    result.meaningless = is_meaningless(raw)

    if _hit(raw, HANDOFF_INTENT_PATTERNS):
        return IntentResult(
            **{**result.__dict__, "handoff": True, "forced": True, "reason": "客户主动要求转人工"}
        )

    forced = _hit(raw, FORCED_HANDOFF_PATTERNS)
    if forced:
        return IntentResult(
            **{**result.__dict__, "handoff": True, "forced": True, "reason": f"高风险话题：{forced}"}
        )

    risk = _hit(raw, RISK_HINT_PATTERNS)
    if risk and _hit(raw, NEGATIVE_PATTERNS):
        return IntentResult(
            **{
                **result.__dict__,
                "handoff": True,
                "forced": True,
                "reason": f"高风险话题（{risk}）+ 负面情绪",
            }
        )

    return result


# --------------------------------------------------------------------------- #
# 置信度
# --------------------------------------------------------------------------- #
def confidence_from_score(top_score: float) -> int:
    """把检索相似度映射成 0-100 的置信度。

    这是启发式映射，不是概率。阈值由 AI 员工配置（confidence_threshold）决定，
    后续换成模型打分时只改本函数。
    """
    if top_score <= 0:
        return 0
    if top_score >= 0.95:
        return 99
    return max(0, min(99, int(round(top_score * 100))))
