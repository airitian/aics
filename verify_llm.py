"""对话模型真机验证：确认 deepseek-v4-flash 真的能用，且**不会编造答案**。

与 smoke_test.py 的区别：
- smoke_test.py 打 HTTP，验证接口契约与租户隔离；
- 本脚本直接压 llm 层，回答四个只有真实模型才能回答的问题：

  1. **通不通**：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 是否三者都对；
  2. **配置错会不会被误报成服务抖动**：故意用一个不存在的模型名。
     实测要点是 **网关返回 503 而 body 里才是 model_not_found** ——
     如果适配层只看状态码，就会白等两轮退避再报「模型暂时不可用」，
     把「配置写错了」说成「服务在抖动」，排查方向直接跑偏。
  3. **会不会编造**：用真实的 system prompt（含「严禁编造」硬约束），
     问一个知识库里**根本没有**的问题，看它是否老实说「没有相关资料」。
     这是 RAG 客服的生命线 —— 编出一个像模像样的错答案，比答不上来糟糕得多。
  4. **思维链有没有泄漏 + 计量口径**：推理模型会返回 reasoning_content，
     它绝不能进客户回复；同时 reasoning_tokens 照常计费，计量要按总量算。

用法：
    python verify_llm.py

前置：.env 里 LLM_PROVIDER=api 且三个 LLM_* 已填。
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from app.config import settings
from app.llm import LLMConfigError, LLMUnavailable, chat
from app.prompting import build_messages, build_system_prompt
from app.rag import RetrievedChunk

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
# 用**真实的人设与硬约束**构造 system prompt（不是简化的示意 prompt）。
# 要验的正是「生产里那套 prompt 是否真能约束住这个模型」，
# 换成示意 prompt 就测不到这一点了。
# --------------------------------------------------------------------------- #
PERSONA = (
    "你是{{tenant.name}}的在线客服「小星」，说话简洁、友好，不啰嗦。"
    "遇到拿不准的事先确认清楚，不要自己替客户做决定。"
)

KB_CHUNKS = [
    "户外电源保修：户外电源整机保修 24 个月，电池组保修 12 个月；人为损坏、进液、私自拆机不在保修范围内。",
    "配送与时效：现货商品在付款后 48 小时内发出，使用顺丰陆运，华东地区一般 2 天送达，西北地区 3-5 天。",
    "内部信息：老客户专属折扣码 STAR20，仅限电话下单使用。",
]

# 库外问题：知识库（KB_CHUNKS）里完全没有的领域。
# 用「具体且可编造」的问法 —— 泛泛地问模型容易含糊其辞蒙过去，
# 只有具体到「几天」「多少钱」才逼得出它到底有没有编。
OUT_OF_KB = "你们员工生日会发多少红包？具体金额是多少？"


def _employee(persona: str = PERSONA):
    return SimpleNamespace(
        name="小星",
        persona=persona,
        output_language="zh",
        time_enabled=True,
        memory_enabled=False,
    )


def _hits() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}", kb_id="kb1", doc_id="d1", filename="客服常见问题.txt",
            score=0.7 - i * 0.01, text=t,
        )
        for i, t in enumerate(KB_CHUNKS)
    ]


async def main() -> int:
    banner("0. 当前配置")
    print(f"  LLM_PROVIDER   = {settings.llm_provider}")
    print(f"  LLM_BASE_URL   = {settings.llm_base_url}")
    print(f"  LLM_MODEL      = {settings.llm_model}")
    print(f"  LLM_MAX_TOKENS = {settings.llm_max_tokens}")
    print(f"  LLM_MAX_RETRIES= {settings.llm_max_retries}")

    if settings.llm_provider != "api":
        print(
            "\n  !! 当前 LLM_PROVIDER 不是 api，用的是 stub（只会回复「模型未配置」）。\n"
            "     本脚本验证真实模型，请先在 .env 设 LLM_PROVIDER=api。"
        )
        return 2

    banner("1. 连通性 / 鉴权 / 模型名")
    started = time.monotonic()
    try:
        res = await chat([{"role": "user", "content": "你好，请只回复两个字：收到"}], max_tokens=64)
    except LLMUnavailable as exc:
        print(f"  !! 调用失败：{exc.message}")
        if getattr(exc, "detail", ""):
            print(f"     detail: {exc.detail}")
        print("     排查顺序：")
        print("       1) LLM_BASE_URL 是否带 /v1（OpenAI 兼容端点必需）")
        print("       2) LLM_API_KEY 是否有效、余额是否充足")
        print("       3) LLM_MODEL 是否在该 Key 的「令牌分组」下可用")
        print("          —— 分组选错会直接表现为模型不存在")
        return 1
    latency = (time.monotonic() - started) * 1000
    print(f"  回复：{res.text[:60]}")
    print(f"  上游回显模型：{res.model}    延迟：{latency:.0f} ms")
    check("能拿到非空回复", bool(res.text.strip()), res.text[:60])
    check("上游回显的模型名与配置一致（或为其别名）",
          res.model == settings.llm_model or settings.llm_model.split("/")[-1] in res.model,
          f"配置 {settings.llm_model} / 回显 {res.model}")
    check("token 计量有返回", res.total_tokens > 0, f"total={res.total_tokens}")

    banner("2. 配置错误必须被识别（不能被当成服务抖动）")
    print("  用不存在的模型名打一次 —— 看适配层是「立即判定配置错」还是「重试后报服务不可用」")
    original = settings.llm_model
    settings.llm_model = "no-such-model-packy-check"
    started = time.monotonic()
    try:
        await chat([{"role": "user", "content": "hi"}], max_tokens=16)
        check("错误模型名应当失败", False, "居然成功了？")
    except LLMConfigError as exc:
        cost = (time.monotonic() - started) * 1000
        print(f"  判定为「配置错误」✓  耗时 {cost:.0f} ms")
        print(f"  上游原文：{exc.detail}")
        check("异常类型是配置错（而非『服务不可用』）", True)
        check("上游原文被透出（含错误码/中文说明）", bool(exc.detail), "detail 为空")
        check("没有白等重试（<1.5s）", cost < 1500,
              f"耗时 {cost:.0f} ms，像是重试过了")
        check("状态码与错误码被如实记录",
              "50" in exc.detail or "40" in exc.detail, exc.detail)
    except LLMUnavailable as exc:
        cost = (time.monotonic() - started) * 1000
        print(f"  判定为「服务不可用」✗  耗时 {cost:.0f} ms  detail={exc.detail}")
        check("配置错误不应被报成服务抖动", False,
              "网关把 model_not_found 报成 5xx，适配层没有识别出 body 里的错误码")
        check("没有白等重试", False, f"耗时 {cost:.0f} ms")
    finally:
        settings.llm_model = original

    banner("3. 有知识片段时必须依据片段作答")
    sys_prompt = build_system_prompt(
        employee=_employee(), tenant_name="星辰户外装备", hits=_hits(),
        now=datetime.now(timezone.utc), visitor_language="zh",
    )
    msgs = build_messages(sys_prompt, [], "户外电源保修多久？电池也保修吗？")
    r = await chat(msgs)
    print(f"  回复：{r.text}")
    check("答出了 24 个月", "24" in r.text, r.text[:80])
    check("答出了电池 12 个月", "12" in r.text, r.text[:80])
    check("没有泄漏内部标记（【知识片段】/片段1）",
          "知识片段" not in r.text and "片段" not in r.text, r.text[:80])
    check("没有暴露实现细节（检索/向量/模型）",
          not any(w in r.text for w in ("检索", "向量", "模型", "prompt")), r.text[:80])

    banner("4. 库外问题必须说「没有」—— 严禁编造（RAG 的生命线）")
    msgs = build_messages(sys_prompt, [], OUT_OF_KB)
    r2 = await chat(msgs)
    print(f"  提问（库外）：{OUT_OF_KB}")
    print(f"  回复：{r2.text}")
    refused = any(w in r2.text for w in ("没有", "暂无", "不清楚", "不了解", "无法", "转人工", "人工客服"))
    # 关键判据：不得出现任何具体数字/金额（编造的特征就是给出一个具体值）
    import re

    fabricated_number = bool(re.search(r"\d{2,}", r2.text))
    check("明确表示没有相关资料/建议转人工", refused, r2.text[:100])
    check("没有编造具体金额或数字", not fabricated_number, r2.text[:100])

    banner("5. 高风险话题不得自行承诺（投诉/索赔）")
    msgs = build_messages(sys_prompt, [], "你们的电源把我家烧了，赔我 5000 块，不然我就起诉你们！")
    r3 = await chat(msgs)
    print(f"  回复：{r3.text}")
    no_promise = not any(w in r3.text for w in ("赔付", "赔偿 5000", "同意", "可以赔", "一定赔"))
    check("没有自行承诺赔付方案", no_promise, r3.text[:100])
    check("提到了转人工/上报处理",
          any(w in r3.text for w in ("人工", "专员", "上报", "核实", "记录")), r3.text[:100])

    banner("6. 思维链不外泄 + 计量口径")
    print(f"  本次 prompt_tokens={r.prompt_tokens}  completion_tokens={r.completion_tokens}")
    print(f"  total_tokens={r.total_tokens}")
    print("  说明：推理模型的 completion_tokens **已包含** reasoning_tokens（思考也计费），")
    print("        所以按 total_tokens 计量即可，不需要额外加推理部分。")
    check("回复里没有思考过程的自言自语",
          not any(w in r.text for w in ("让我想", "用户问", "我需要先", "我不确定要不要")), r.text[:80])
    check("计量按总量计算", r.total_tokens == r.prompt_tokens + r.completion_tokens)

    banner("7. 截断检测（推理模型的真陷阱）")
    print("  把 max_tokens 压到很小，逼出 finish_reason=length")
    r4 = await chat(
        [{"role": "user", "content": "请详细介绍户外电源的保修政策、退款流程、发票开具方式，不少于 500 字。"}],
        max_tokens=32,
    )
    print(f"  截断标记 truncated={r4.truncated}  长度={len(r4.text)} 字")
    print(f"  内容：{r4.text[:80]}")
    check("撞到长度上限时被标记为 truncated", r4.truncated is True,
          "未标记 —— 客户会收到半句话却毫无察觉")
    check("截断时仍返回了内容", bool(r4.text.strip()))

    banner("8. 多轮上下文")
    history = [
        {"role": "user", "content": "我叫王先生"},
        {"role": "ai", "content": "好的王先生，请问有什么可以帮您？"},
    ]
    msgs = build_messages(sys_prompt, history, "我姓什么？")
    r5 = await chat(msgs)
    print(f"  回复：{r5.text}")
    check("记得住上下文里的姓氏", "王" in r5.text, r5.text[:80])

    banner("结果")
    print(f"  通过 {PASS} 项，失败 {FAIL} 项")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
