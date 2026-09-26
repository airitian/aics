"""提示词构建。

关键：把「严禁编造」「工具失败要说明」「高风险强制转人工」写成**系统级硬约束**，
不依赖人设里有没有写。人设是业务方写的，硬约束必须由系统保证（PRD 十章强约束）。
"""
from __future__ import annotations

import re
from datetime import datetime

from app.config import settings
from app.models import AiEmployee
from app.rag import RetrievedChunk
from app.utils import truncate

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}")


def render_persona(persona: str, variables: dict[str, str] | None = None) -> str:
    """渲染 {{变量}}；单值超限截断加省略号（PRD 3.1.3）。"""
    variables = variables or {}
    limit = settings.persona_var_max_chars

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)  # 保留原样，发布前校验会拦截
        return truncate(str(variables[key]), limit)

    return _VAR_RE.sub(_sub, persona or "")


def extract_variables(persona: str) -> list[str]:
    return sorted({m.group(1) for m in _VAR_RE.finditer(persona or "")})


AVAILABLE_VARIABLES = [
    "tenant.name",
    "visitor.id",
    "visitor.language",
    "channel",
    "now",
    "employee.name",
]


def unknown_variables(persona: str) -> list[str]:
    return [v for v in extract_variables(persona) if v not in AVAILABLE_VARIABLES]


LANGUAGE_NAMES = {
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "ru": "Русский",
    "es": "Español",
    "pt": "Português",
    "fr": "Français",
    "de": "Deutsch",
    "ar": "العربية",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
}


def _format_hits(hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "（本次检索没有任何知识片段）"
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        source = hit.filename or "知识库"
        lines.append(f"[片段{i}] 来源：{source}｜相关度：{hit.score:.2f}\n{hit.text}")
    return "\n\n".join(lines)


def build_system_prompt(
    *,
    employee: AiEmployee,
    tenant_name: str,
    hits: list[RetrievedChunk],
    now: datetime,
    visitor_language: str = "",
    memory_lines: list[str] | None = None,
    tool_note: str = "",
) -> str:
    parts: list[str] = []

    rendered_persona = render_persona(
        employee.persona,
        {
            "tenant.name": tenant_name,
            "employee.name": employee.name,
            "now": now.strftime("%Y-%m-%d %H:%M"),
            "visitor.language": visitor_language or "未知",
        },
    )
    if rendered_persona.strip():
        parts.append("【角色设定】\n" + rendered_persona.strip())

    # ---- 语言 ----
    if employee.output_language == "auto":
        lang_line = (
            "使用与访客相同的语言回复。"
            + (f"当前识别到的访客语言：{visitor_language}。" if visitor_language else "")
        )
    else:
        name = LANGUAGE_NAMES.get(employee.output_language, employee.output_language)
        lang_line = f"必须使用 {name} 回复，无论访客使用什么语言。"
    parts.append("【输出语言】\n" + lang_line)

    # ---- 时间 ----
    if employee.time_enabled:
        parts.append(f"【当前时间】\n{now.strftime('%Y-%m-%d %H:%M')}（时区：{tenant_name} 项目时区）")
    else:
        parts.append("【当前时间】\n未开启。若访客询问时间或涉及日期的推算，必须说明无法提供，不得凭训练数据猜测。")

    # ---- 记忆 ----
    if employee.memory_enabled and memory_lines:
        parts.append("【已知客户信息】\n" + "\n".join(f"- {m}" for m in memory_lines))

    # ---- 拟人化回复 ----
    if getattr(employee, "humanize_enabled", False):
        parts.append(
            "【拟人化回复】\n"
            "你现在以真人客服的身份与客户聊天，注意：\n"
            "1. 语气口语化、自然亲切，像真人打字；避免「您好，很高兴为您服务」「感谢您的理解」等机械客服用语。\n"
            "2. 回复简短，一次说一两件事，不要长篇大论；可以适当分段。\n"
            "3. 先回应客户情绪（如需要），再给方案，最后自然引导下一步。\n"
            "4. 偶尔可用语气词（啦/呀/～）让表达更自然，但连续两句不要都用。\n"
            "5. 无论如何不得违反「必须遵守的规则」：查不到的仍然如实说查不到，不编造；该转人工的仍然转人工。"
        )

    # ---- 知识片段 ----
    parts.append("【知识片段】\n" + _format_hits(hits))

    # ---- 对话上下文 ----
    # 硬规则第 1 条要求"只能依据知识片段作答"，模型会把访客在会话里说过的话
    # 也判成"没有资料"，进而声称自己"没有记录功能"——明明历史就在消息数组里。
    # 这里显式授权：访客说过的话可以直接引用，业务事实仍以知识片段为准。
    parts.append(
        "【对话上下文】\n"
        "你可以看到本会话此前的完整对话记录。访客在对话中主动提供的信息"
        "（姓名、称呼、订单号、会员号、设备型号、描述过的问题等），"
        "你可以直接引用、确认和延续，这属于正常记忆而非编造；"
        "但业务事实类答案（政策、参数、价格、时效等）仍必须以【知识片段】为准。\n"
        "不要声称自己「没有记录功能」「无法保存」「记不住」——本会话内的信息你能正常记住并使用。"
        "只有当访客要求把信息保存到下次咨询、会员账户等跨会话场景时，才如实说明跨会话无法保留。"
    )

    if tool_note:
        parts.append("【工具调用情况】\n" + tool_note)

    # ---- 硬约束（系统级，不可被角色设定覆盖） ----
    parts.append(
        """【必须遵守的规则】
1. 只能依据上面的【知识片段】作答。片段为空、或片段中不含答案时，必须明确告诉访客"我这里暂时没有相关资料"，并询问是否需要转人工，**严禁编造、推测或使用你的通用知识作答**。访客在本对话中已提供的信息可直接引用，不受此条限制。
2. 不得引用"【知识片段】【片段1】"等内部标记，也不要提到检索、向量、模型等实现细节。
3. 涉及投诉、纠纷、赔偿、法律追责，以及讨价还价类诉求时，不要自行承诺任何处理方案或赔付条件，直接说明会为其转接人工客服。
4. 工具调用失败或未返回结果时，必须说明"暂时无法查询"，不得编造查询结果。
5. 回复简洁自然，像真人客服；不确定时先澄清，不要连续追问超过 2 次。
6. 不谈论与本业务无关的话题；遇到辱骂或灌水保持礼貌并把话题拉回业务。
7. 资料里的字段值若明显与字段含义不符（例如"Frequency"栏填成存储容量、"价格"栏填成日期），
   **不得自行推断或套用同类产品的常见值**：如实说明资料中记录的该值，指出它可能填错了，
   并建议访客找人工核实。宁可说"资料这里可能填错了"，也不要给一个看似合理但无据的数字。
8. 故障排查类问题：资料里列出的**全部**可能原因与对应解决方法都要完整列出（逐条编号），
   不得只挑其中几条就结束；片段中没有的内容不得自行补充。
9. 安全红线：资料中标注仅适用于特定场景的操作步骤（如「废弃本机时」的电池拆卸步骤），
   **绝不能**提供给想自行维修、更换配件、拆解产品的访客。这类诉求必须明确告知
   禁止自行操作，并建议联系售后服务人员或专业人员处理。
10. 访客询问某功能"能不能/支不支持"（如语音控制、某种 Wi-Fi、某种地毯）时，
    只要片段里有对应说明，必须**先给出明确的支持/不支持结论**，再补充限定条件；
    不得不作答、只反问澄清。资料未提及的能力如实说明"资料未写明"。
11. 访客表示发来了图片、而对话里没有图片内容时，如实说明你无法查看图片，
    请访客用文字描述问题或零件名称；不得说"没查到资料"，更不得假装看到了图片。
12. 资料未收录的信息（如价格、保修期限、促销活动）如实说明未收录，并建议通过
    官方商城、官网或售后热线等官方渠道查询；部件的清洁/更换周期必须与该部件
    严格一一对应，**不得把一个部件的周期安到另一个部件上**。滤芯/过滤件等
    耗材只属于原文写明的所属部件（如尘盒、清洁槽底板）；当用户问的部件在
    原文中没有滤芯条目时，应明确说明该部件资料中未提及滤芯，再把滤芯周期
    归回其所属部件，不要顺着用户的问题默认"也有滤芯"。
13. 保养、存放、停用、使用注意事项类问题：逐条核对上下文要点，**完整覆盖全部
    前置动作与周期性要求**（如"先充满电再关机""每 1.5 个月补充电"这类步骤和
    周期），不得只答其中一条（如只说"断电"）就收尾；同一段资料在不同章节
    重复出现的要求按最完整的一版作答。
14. 你没有任何工具可调用：**不得输出工具调用标记或内部协议文本**
    （如 DSML、<calls>、<invoke> 等），一律直接用自然语言回复；也不要输出
    任何与对话无关的控制字符或占位符。
15. 回复中建议访客"联系售后服务人员/联系售后"时，若资料中给出了官方
    售后热线等联系方式，**必须同时写明该号码**，不得只说"联系售后"。
16. 访客询问某部件多久更换一次、而资料中该部件**只有清洁/清理要求、
    没有更换条目**时，必须明确说明"该部件无需更换，仅需按周期清洁"，
    不得只说"资料中没有相关条目"就带过。
17. 涉及长期存放、停用、电池过放等场景时，资料中的**安全警告必须完整
    复述**（如"电池过放或长期不用可能无法充电，需联系售后维修，切勿
    自行拆卸"），不得省略警告部分只答操作步骤。
18. 访客明确要求特定输出格式（如"用 JSON 输出，字段为 X 和 Y"）时，
    **严格按访客要求的结构输出**：要数组就给数组，字段名用访客指定的，
    不得自行包一层对象、改名或附加额外字段；格式服从访客，内容仍只
    依据资料。
19. 访客描述模糊（如"机器不行了""用不了"）需要澄清时，按固定维度
    **一次性列全追问**：① 具体现象/表现 ② 产品型号 ③ 已尝试过的操作
    ④ App 或机器的报错提示；不得只问其中一两项就开始排查。"""
    )
    return "\n\n".join(parts)


def build_messages(
    system_prompt: str, history: list[dict], user_message: str, max_history: int = 40
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for item in history[-max_history:]:
        role = "assistant" if item.get("role") in ("ai", "agent") else "user"
        content = (item.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


# 拆分触发的最小长度：文档口径「只会在回答内容较长时触发」，短回答硬拆反而奇怪
SPLIT_MIN_CHARS = 80


def split_reply_segments(text: str, max_segments: int) -> list[str]:
    """把较长回复拆成多条消息。

    规则：优先按空行分段（模型最自然的分段信号）；没有空行时按句子切分。
    段数超过 max_segments 时把多余内容并进最后一段；无论如何不产生空段。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= SPLIT_MIN_CHARS:
        return [text]
    max_segments = max(2, min(8, int(max_segments or 2)))

    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) < 2:
        sentences = [s.strip() for s in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text) if s.strip()]
        if len(sentences) < 2:
            return [text]
        # 均匀切成 2..max_segments 组
        n_groups = min(max_segments, len(sentences) - 1) if len(sentences) <= max_segments * 2 else max_segments
        # 让每组至少 1 句，且组数不超过句子数
        n_groups = max(2, min(n_groups, len(sentences)))
        base, extra = divmod(len(sentences), n_groups)
        parts, i = [], 0
        for g in range(n_groups):
            size = base + (1 if g < extra else 0)
            parts.append("".join(sentences[i : i + size]).strip())
            i += size
        parts = [p for p in parts if p]
    if len(parts) <= max_segments:
        return parts
    head = parts[: max_segments - 1]
    tail = "\n".join(parts[max_segments - 1 :])
    return head + [tail]


NO_ANSWER_REPLY = "我这边暂时没有查到相关资料，您可以换个说法再问一次，或者我帮您转接人工客服，可以吗？"
HANDOFF_REPLY = "好的，我这就为您转接人工客服，请稍等。"
HANDOFF_QUEUED_REPLY = "已为您记录，当前人工客服不在线，我们会尽快通过留言或您留下的联系方式回复您。"
DEGRADED_REPLY = "抱歉，我暂时无法处理您的消息，已为您转人工，请稍候。"
CLARIFY_REPLY = "抱歉，我没太理解您的意思，能麻烦您再说得具体一点吗？比如您想咨询的是产品、订单还是售后问题？"
