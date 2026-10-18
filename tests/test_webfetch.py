"""网页内容提取（Fetch）测试。

覆盖四条线：
1. URL 提取：格式边界、中文标点、去重、上限、无协议不抓；
2. SSRF 防护：环回/内网/链路本地/元数据地址/非 http 协议/重定向跳内网；
3. 正文提取：script/style 剥离、title/meta description、块级换行、截断；
4. agent 集成：注入 llm_message 与检索查询、开关关闭不抓、
   有网页内容时跳过「无知识命中」兜底（核心场景不能转人工）。
"""
from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from app import webfetch
from app.webfetch import (
    FetchedPage,
    build_note,
    build_query_extra,
    extract_urls,
    fetch_for_message,
    fetch_url,
    has_content,
    html_to_text,
)


# --------------------------------------------------------------------------- #
# 1. URL 提取
# --------------------------------------------------------------------------- #

class TestExtractUrls:
    def test_single(self):
        assert extract_urls("帮我看看 https://shop.example.com/item/123 是不是你家的") == [
            "https://shop.example.com/item/123"
        ]

    def test_multiple_keep_order_and_dedupe(self):
        text = "第一个 https://a.com/1 第二个 https://b.com/2 还有 https://a.com/1"
        assert extract_urls(text) == ["https://a.com/1", "https://b.com/2"]

    def test_limit(self):
        text = " ".join(f"https://s{i}.com/x" for i in range(5))
        assert len(extract_urls(text, limit=3)) == 3

    def test_chinese_punctuation_boundary(self):
        text = "看这个链接：https://item.jd.com/100012043978.html，是不是你们的产品？"
        assert extract_urls(text) == ["https://item.jd.com/100012043978.html"]

    def test_markdown_link(self):
        assert extract_urls("[商品](https://a.com/x)") == ["https://a.com/x"]

    def test_trailing_punct_stripped(self):
        assert extract_urls("https://a.com/x。）") == ["https://a.com/x"]

    def test_http_and_case(self):
        assert extract_urls("HTTP://A.COM/X") == ["HTTP://A.COM/X"]

    def test_no_url(self):
        assert extract_urls("这个商品多少钱") == []

    def test_bare_domain_not_extracted(self):
        # 无协议写法刻意不抓：猜错成本比漏抓高
        assert extract_urls("www.example.com 这个是你家的吗") == []

    def test_empty(self):
        assert extract_urls("") == []
        assert extract_urls(None) == []


# --------------------------------------------------------------------------- #
# 2. SSRF 防护
# --------------------------------------------------------------------------- #

class TestSsrf:
    @pytest.mark.parametrize("host", [
        "127.0.0.1", "localhost", "10.0.0.5", "192.168.1.1",
        "172.16.0.1", "169.254.169.254", "0.0.0.0", "[::1]",
    ])
    def test_blocked_targets(self, monkeypatch, host):
        # 直接给 IP 字面量，绕过 DNS 也逃不掉地址段校验
        url = f"http://{host}/admin"
        ok, reason = webfetch._validate_target(url)
        assert ok is False, f"{host} 本应被拒"
        assert reason

    def test_blocked_schemes(self):
        for url in ["ftp://example.com/x", "javascript:alert(1)",
                    "file:///etc/passwd", "data:text/html,hi"]:
            ok, reason = webfetch._validate_target(url)
            assert ok is False, url

    def test_dns_private_ip_blocked(self, monkeypatch):
        # 公网域名解析到内网 IP（DNS rebinding / 内部域）→ 拒
        def fake_getaddrinfo(host, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))]
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)
        ok, reason = webfetch._validate_target("http://evil.example.com/x")
        assert ok is False
        assert "不允许访问" in reason

    def test_dns_public_ip_allowed(self, monkeypatch):
        def fake_getaddrinfo(host, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)
        ok, _ = webfetch._validate_target("http://example.com/x")
        assert ok is True

    def test_unresolvable_blocked(self, monkeypatch):
        def boom(host, *a, **kw):
            raise socket.gaierror("nope")
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", boom)
        ok, reason = webfetch._validate_target("http://nope.invalid/x")
        assert ok is False
        assert "无法解析" in reason

    def test_fake_ip_allowed_only_with_proxy(self, monkeypatch):
        """198.18.0.0/15（Clash fake-ip 段）只在配了代理时放行——生产 DNS 下
        域名不会解析到该段；不配代理时必须照拦，否则白名单变后门。"""
        def fake_getaddrinfo(host, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.240", 80))]
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(webfetch, "_HAS_PROXY", True)
        assert webfetch._validate_target("http://shop.example.com/x")[0] is True
        monkeypatch.setattr(webfetch, "_HAS_PROXY", False)
        ok, reason = webfetch._validate_target("http://shop.example.com/x")
        assert ok is False
        assert "不允许访问" in reason

    def test_redirect_to_internal_blocked(self, monkeypatch):
        # 公网 URL 302 跳 169.254.169.254（云元数据）→ 第二跳被拒
        def fake_getaddrinfo(host, *a, **kw):
            ip = "169.254.169.254" if "internal" in host else "93.184.216.34"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)

        def handler(request: httpx.Request) -> httpx.Response:
            if "internal" in str(request.url):
                return httpx.Response(200, text="metadata-secret")
            return httpx.Response(302, headers={"Location": "http://internal/latest/meta-data"})

        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return original(*args, transport=transport, **kwargs)

        monkeypatch.setattr(webfetch.httpx, "AsyncClient", client_factory)
        page = asyncio.run(fetch_url("http://public.example.com/go"))
        assert page.ok is False
        assert "不被允许" in page.error
        assert "metadata-secret" not in page.text

    def test_redirect_chain_limit(self, monkeypatch):
        def fake_getaddrinfo(host, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "http://example.com/loop"})

        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return original(*args, transport=transport, **kwargs)

        monkeypatch.setattr(webfetch.httpx, "AsyncClient", client_factory)
        page = asyncio.run(fetch_url("http://example.com/loop"))
        assert page.ok is False
        assert "重定向次数过多" in page.error


# --------------------------------------------------------------------------- #
# 3. 正文提取
# --------------------------------------------------------------------------- #

class TestHtmlToText:
    def test_basic(self):
        html = """<html><head><title>TX-500 无线网桥 - 示例商城</title>
        <meta name="description" content="室外高功率网桥，传输距离 5 公里">
        <style>.x{color:red}</style></head>
        <body><nav>首页 | 登录</nav>
        <h1>TX-500 无线网桥</h1>
        <p>售价 <b>¥899</b>，支持 5.8G 频段。</p>
        <script>var a = 1 < 2;</script>
        <div>保修两年</div>
        </body></html>"""
        title, text, truncated = html_to_text(html)
        assert title == "TX-500 无线网桥 - 示例商城"
        assert "首页 | 登录" not in text  # nav 不算正文……其实保留也无妨，但不该有 style 内容
        assert ".x{color:red}" not in text
        assert "var a = 1" not in text
        assert "TX-500 无线网桥" in text
        assert "¥899" in text
        assert "5.8G 频段" in text
        assert "保修两年" in text
        assert "室外高功率网桥" in text  # meta description 已并入正文
        assert truncated is False

    def test_entities_decoded(self):
        _, text, _ = html_to_text("<p>A &amp; B &lt;tag&gt; &#20013;</p>")
        assert "A & B <tag> 中" in text

    def test_truncation(self):
        _, text, truncated = html_to_text("<p>" + "长" * 100 + "</p>", max_chars=10)
        assert truncated is True
        assert len(text) == 10

    def test_empty(self):
        _, text, _ = html_to_text("<html><body></body></html>")
        assert text.strip() == ""

    def test_malformed_html_does_not_crash(self):
        title, text, _ = html_to_text("<p>未闭合 <b>粗体 <div>混叠")
        assert "未闭合" in text


# --------------------------------------------------------------------------- #
# 4. 注入块构造
# --------------------------------------------------------------------------- #

class TestBuildNote:
    def test_ok_page(self):
        pages = [FetchedPage(url="https://a.com/x", ok=True, title="商品A",
                             text="品牌：示例；型号：TX-500")]
        note = build_note(pages)
        assert "网页1" in note
        assert "商品A" in note
        assert "TX-500" in note
        assert "外部资料" in note  # 防注入声明必须在

    def test_failed_page_tells_model_to_say_unopenable(self):
        pages = [FetchedPage(url="https://a.com/x", ok=False, error="访问超时")]
        note = build_note(pages)
        assert "抓取失败" in note
        assert "访问超时" in note
        assert "链接暂时打不开" in note
        assert "不要让访客重新表述问题" in note

    def test_mixed(self):
        pages = [
            FetchedPage(url="https://a.com/1", ok=True, title="A", text="内容A"),
            FetchedPage(url="https://a.com/2", ok=False, error="域名无法解析"),
        ]
        note = build_note(pages)
        assert "网页1" in note and "网页2" in note
        assert "内容A" in note and "抓取失败" in note

    def test_empty(self):
        assert build_note([]) == ""

    def test_truncated_marker(self):
        pages = [FetchedPage(url="https://a.com", ok=True, text="x", truncated=True)]
        assert "已截断" in build_note(pages)

    def test_query_extra_only_ok_pages(self):
        pages = [
            FetchedPage(url="https://a.com/1", ok=True, title="无线网桥", text="TX-500 参数……" * 200),
            FetchedPage(url="https://a.com/2", ok=False, error="超时"),
        ]
        extra = build_query_extra(pages)
        assert "无线网桥" in extra
        assert "TX-500" in extra
        assert len(extra) < 3000  # 短版：不能把整页塞进检索查询

    def test_has_content(self):
        assert has_content([FetchedPage(url="u", ok=True, text="x")]) is True
        assert has_content([FetchedPage(url="u", ok=False, error="e")]) is False
        assert has_content([FetchedPage(url="u", ok=True, text="  ")]) is False
        assert has_content([]) is False


# --------------------------------------------------------------------------- #
# 5. fetch_url / fetch_for_message（MockTransport，不真连外网）
# --------------------------------------------------------------------------- #

_HTML = """<html><head><title>示例商品</title></head>
<body><h1>TX-500</h1><p>这是一段商品描述。</p></body></html>"""


def _patch_transport(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return original(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(webfetch.httpx, "AsyncClient", factory)


def _patch_dns(monkeypatch, ip="93.184.216.34"):
    def fake_getaddrinfo(host, *a, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]
    monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)


class TestFetchUrl:
    def test_success(self, monkeypatch):
        _patch_dns(monkeypatch)
        _patch_transport(monkeypatch, lambda req: httpx.Response(
            200, html=_HTML, headers={"content-type": "text/html; charset=utf-8"}))
        page = asyncio.run(fetch_url("https://shop.example.com/item/1"))
        assert page.ok is True
        assert page.title == "示例商品"
        assert "TX-500" in page.text
        assert page.status_code == 200
        assert page.elapsed_ms > 0

    def test_http_error_status(self, monkeypatch):
        _patch_dns(monkeypatch)
        _patch_transport(monkeypatch, lambda req: httpx.Response(404, html="not found"))
        page = asyncio.run(fetch_url("https://example.com/missing"))
        assert page.ok is False
        assert "404" in page.error

    def test_non_textual_rejected(self, monkeypatch):
        _patch_dns(monkeypatch)
        _patch_transport(monkeypatch, lambda req: httpx.Response(
            200, content=b"\x89PNG\r\n", headers={"content-type": "image/png"}))
        page = asyncio.run(fetch_url("https://example.com/a.png"))
        assert page.ok is False
        assert "不是网页内容" in page.error

    def test_no_text_content(self, monkeypatch):
        _patch_dns(monkeypatch)
        _patch_transport(monkeypatch, lambda req: httpx.Response(
            200, html="<html><body><div></div></body></html>",
            headers={"content-type": "text/html"}))
        page = asyncio.run(fetch_url("https://example.com/empty"))
        assert page.ok is False
        assert "没有可提取的文字" in page.error

    def test_relative_redirect_followed(self, monkeypatch):
        _patch_dns(monkeypatch)

        def handler(req: httpx.Request):
            if str(req.url).endswith("/go"):
                return httpx.Response(302, headers={"Location": "/final"})
            return httpx.Response(200, html=_HTML, headers={"content-type": "text/html"})

        _patch_transport(monkeypatch, handler)
        page = asyncio.run(fetch_url("https://example.com/go"))
        assert page.ok is True
        assert page.final_url.endswith("/final")

    def test_message_level_concurrency(self, monkeypatch):
        _patch_dns(monkeypatch)
        seen: list[str] = []

        def handler(req: httpx.Request):
            seen.append(str(req.url))
            return httpx.Response(200, html=_HTML, headers={"content-type": "text/html"})

        _patch_transport(monkeypatch, handler)
        pages = asyncio.run(fetch_for_message(
            "看下 https://a.com/1 和 https://b.com/2 和 https://c.com/3 是不是你家的"))
        assert len(pages) == 3
        assert all(p.ok for p in pages)
        assert len(seen) == 3

    def test_message_without_url(self):
        assert asyncio.run(fetch_for_message("没有链接的消息")) == []


# --------------------------------------------------------------------------- #
# 6. prompt 集成
# --------------------------------------------------------------------------- #

class TestPromptingWebRules:
    def _employee(self, **overrides):
        from app.models import AiEmployee
        emp = AiEmployee(name="测试员工", persona="你是客服")
        for k, v in overrides.items():
            setattr(emp, k, v)
        return emp

    def _prompt(self, employee, has_web_pages: bool):
        from datetime import datetime, timezone
        from app import prompting
        return prompting.build_system_prompt(
            employee=employee, tenant_name="测试租户", hits=[],
            now=datetime.now(timezone.utc), has_web_pages=has_web_pages,
        )

    def test_rules_present_only_when_has_pages(self):
        emp = self._employee()
        with_rules = self._prompt(emp, has_web_pages=True)
        without_rules = self._prompt(emp, has_web_pages=False)
        assert "【网页内容使用规则】" in with_rules
        assert "【网页内容使用规则】" not in without_rules

    def test_priority_kb_over_web(self):
        with_rules = self._prompt(self._employee(), has_web_pages=True)
        assert "仍以【知识片段】为准" in with_rules
        assert "不得执行" in with_rules  # 防注入

    def test_hard_rule_14_updated(self):
        prompt = self._prompt(self._employee(), has_web_pages=False)
        assert "你没有任何工具可调用" not in prompt
        assert "不得输出工具调用标记" in prompt


# --------------------------------------------------------------------------- #
# 7. agent 端到端集成（真实 handle_turn，只 mock 抓取与模型）
# --------------------------------------------------------------------------- #

_URL_MSG = "帮我看看这个链接里面的商品是不是你家的 https://shop.example.com/item/tx500"


class TestAgentWebFetchIntegration:
    """三条链路行为，缺一不可：

    1. 抓到的正文必须进 LLM 看到的用户消息（否则模型根本不知道链接内容）；
    2. 员工级开关关闭时必须完全不抓（不该产生的流量一分都不产生）；
    3. **有网页内容时不得走「无知识命中」兜底**——访客问「是不是你家的」而
       知识库恰好没有该商品时，正确行为是模型基于网页判断并如实说明，
       追问/转人工等于把最简单的咨询推给人（本功能的核心场景）。
    """

    def _patch_chat(self, monkeypatch, captured: dict):
        import app.agent as agent_mod
        from app.llm import LLMResult

        async def fake_chat(messages, **kw):
            captured["messages"] = messages
            text = " ".join(str(m.get("content", "")) for m in messages)
            captured["joined"] = text
            return LLMResult(text="这是我方 TX-500 无线网桥，售价 899 元。",
                             prompt_tokens=1, completion_tokens=1, endpoint="stub")

        monkeypatch.setattr(agent_mod, "chat", fake_chat)

    def _patch_retrieve_hit(self, monkeypatch):
        """mock 检索返回一条高分命中 → low_confidence=False → 必走主生成路径。

        不能依赖演示知识库与兜底层行为：全量跑时前面的用例会改员工/知识库
        状态，无命中的轮次可能走 lowconf_clarify（根本不调模型，captured 为空），
        那是兜底链的测试范畴，不是这里要断言的。
        """
        import app.agent as agent_mod
        from app.rag import RetrievedChunk

        async def fake_retrieve(db, **kw):
            return [RetrievedChunk(
                chunk_id="c-test", kb_id="kb-test", doc_id="doc-test",
                text="TX-500 无线网桥：售价 899 元，支持 2.4G/5G 双频。",
                filename="产品参数.pdf", score=0.9,
            )], 1

        monkeypatch.setattr(agent_mod, "retrieve", fake_retrieve)

    def _patch_fetch(self, monkeypatch, pages):
        import app.agent as agent_mod

        async def fake_fetch(message, **kw):
            return pages

        monkeypatch.setattr(agent_mod.webfetch, "fetch_for_message", fake_fetch)

    def _send(self, client, key, message, visitor):
        return client.post(
            "/api/chat/message", headers={"X-Widget-Key": key},
            json={"message": message, "visitor_id": visitor},
        ).json()

    def _enable(self, client, star, emp_id):
        """每个用例开头显式置回开启——员工是跨用例共享的同一行，
        上一个开关用例的 False 会污染后面的用例（实测踩过）。"""
        client.patch(f"/api/employees/{emp_id}", headers=star,
                     json={"web_fetch_enabled": True})

    def test_page_content_reaches_model(self, client, star, monkeypatch):
        from tests.test_flows import _default_employee, _widget_key

        emp = _default_employee(client, star)
        self._enable(client, star, emp["id"])
        key = _widget_key(client, star, emp["id"])
        captured: dict = {}
        self._patch_chat(monkeypatch, captured)
        self._patch_retrieve_hit(monkeypatch)
        self._patch_fetch(monkeypatch, [
            FetchedPage(url="https://shop.example.com/item/tx500", ok=True,
                        title="TX-500 无线网桥", text="品牌：示例；型号：TX-500；售价 ¥899")
        ])

        reply = self._send(client, key, _URL_MSG, "v-webfetch-ok")
        assert reply["reply"] == "这是我方 TX-500 无线网桥，售价 899 元。"
        assert "TX-500 无线网桥" in captured["joined"]
        assert "https://shop.example.com/item/tx500" in captured["joined"]
        system = captured["messages"][0]["content"]
        assert "【网页内容使用规则】" in system
        assert "外部资料" in captured["joined"]  # 防注入声明随内容一起进 prompt

    def test_employee_switch_off_skips_fetch(self, client, star, monkeypatch):
        from tests.test_flows import _default_employee, _widget_key

        emp = _default_employee(client, star)
        client.patch(f"/api/employees/{emp['id']}", headers=star,
                     json={"web_fetch_enabled": False})
        key = _widget_key(client, star, emp["id"])

        import app.agent as agent_mod
        calls = {"n": 0}

        async def counting_fetch(message, **kw):
            calls["n"] += 1
            return []

        monkeypatch.setattr(agent_mod.webfetch, "fetch_for_message", counting_fetch)
        captured: dict = {}
        self._patch_chat(monkeypatch, captured)
        self._patch_retrieve_hit(monkeypatch)

        self._send(client, key, _URL_MSG, "v-webfetch-off")
        assert calls["n"] == 0, "员工开关关闭时一次抓取都不该发生"
        # 系统提示里出现「网页」二字是硬约束 14 的说明文案，不算注入；
        # 要断言的是**访客消息体**里没有网页块
        assert "外部资料" not in captured["messages"][-1]["content"]
        assert "[网页1]" not in captured["messages"][-1]["content"]

    def test_web_content_bypasses_low_confidence_fallback(self, client, star, monkeypatch):
        """核心场景：知识库没有该商品 + 链接抓取成功 → 必须让模型答，不追问不转人工。"""
        from tests.test_flows import _default_employee, _widget_key

        emp = _default_employee(client, star)
        self._enable(client, star, emp["id"])
        # 收紧兜底策略：一旦走低置信分支，第 1 轮就会回固定话术/转人工
        client.patch(f"/api/employees/{emp['id']}", headers=star,
                     json={"low_confidence_policy": "clarify", "clarify_rounds": 1})
        key = _widget_key(client, star, emp["id"])

        # 对照组：抓取失败（无网页内容）→ 冷门问题无知识命中 → 必须走兜底路径。
        # 用 degrade_reason 断言路径而非比对回复文本：兜底层让桩模型"照着概览
        # 硬凑一句"正是线上真实行为，回复文本什么都证明不了。
        self._patch_fetch(monkeypatch, [
            FetchedPage(url="https://shop.example.com/item/zzz", ok=False, error="访问超时")
        ])
        captured: dict = {}
        self._patch_chat(monkeypatch, captured)
        control = self._send(client, key, "帮我看看 https://shop.example.com/item/zzz 这个是什么",
                             "v-webfetch-ctrl")
        assert control.get("degrade_reason"), (
            f"抓取失败却没有走任何兜底路径：{control}"
        )

        # 实验组：同样冷门问题，抓取成功 → 直接生成，不追问不转人工、无降级标记
        self._patch_fetch(monkeypatch, [
            FetchedPage(url="https://shop.example.com/item/tx500", ok=True,
                        title="冷门商品", text="某个知识库里绝对没有的商品参数")
        ])
        reply = self._send(client, key, "帮我看看 https://shop.example.com/item/tx500 是不是你家的",
                           "v-webfetch-exp")
        assert reply["reply"] == "这是我方 TX-500 无线网桥，售价 899 元。", (
            f"有网页内容却走了兜底分支：{reply}"
        )
        assert not reply.get("degrade_reason"), (
            f"有网页内容仍被标记降级：{reply.get('degrade_reason')}"
        )
        assert reply["handoff"] is False
        assert reply["status"] not in ("queued",)

    def test_fetch_failure_message_reaches_model(self, client, star, monkeypatch):
        """抓失败时模型必须能告诉访客「链接打不开」。"""
        from tests.test_flows import _default_employee, _widget_key

        emp = _default_employee(client, star)
        self._enable(client, star, emp["id"])
        key = _widget_key(client, star, emp["id"])
        captured: dict = {}
        self._patch_chat(monkeypatch, captured)
        self._patch_retrieve_hit(monkeypatch)
        self._patch_fetch(monkeypatch, [
            FetchedPage(url="https://dead.example.com/x", ok=False, error="域名无法解析")
        ])

        self._send(client, key, "看看 https://dead.example.com/x 里是什么", "v-webfetch-dead")
        assert "抓取失败" in captured["joined"]
        assert "域名无法解析" in captured["joined"]
        assert "不要让访客重新表述问题" in captured["joined"]
