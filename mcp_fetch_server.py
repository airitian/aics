#!/usr/bin/env python
"""Fetch MCP server —— 把 AICS 的网页抓取能力暴露成标准 MCP 工具。

用途两条路，互不影响：
1. AICS 自己在 handle_turn 里直接调 app/webfetch.py（默认路径，
   「检测 URL → 抓取 → 注入」的确定性管道，不走 tool-calling 协议）；
2. 任何 MCP 客户端（Claude Desktop、Cherry Studio 等）把本文件配成
   stdio server，就能拿到 `fetch` 工具抓网页。

协议：MCP over stdio = JSON-RPC 2.0，一行一个消息（无 Content-Length 头）。
这里只实现本工具需要的三个方法：initialize / tools/list / tools/call，
刻意不引 `mcp` SDK——协议本身十几行，为一个工具拖一套依赖不值。

运行：
    python mcp_fetch_server.py            # stdio 模式
客户端配置示例（Claude Desktop / claude_desktop_config.json）：
    {"mcpServers": {"fetch": {"command": "python",
     "args": ["C:/path/to/aics/mcp_fetch_server.py"]}}}

安全：app/webfetch.py 里的 SSRF 防护（协议白名单 / DNS 后地址段校验 /
重定向逐跳复检 / 大小与超时上限）对这条路同样生效——入口不同，闸门同一道。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 允许从任意 cwd 启动：把项目根（本文件所在目录）加进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import webfetch  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "aics-fetch", "version": "1.0.0"}

TOOLS = [
    {
        "name": "fetch",
        "description": (
            "抓取一个网页并提取正文纯文本（title + 可读正文，去 script/style）。"
            "用于「看看这个链接里是什么 / 这个商品是不是某家的」这类需要读网页内容的问题。"
            "只支持 http/https；内网与环回地址会被拒绝。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页地址（http/https）"},
                "max_chars": {
                    "type": "integer",
                    "description": "返回正文的最大字符数，默认 6000",
                    "default": 6000,
                },
            },
            "required": ["url"],
        },
    }
]


def _result(msg_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


async def _call_fetch(arguments: dict) -> dict:
    url = str(arguments.get("url", "")).strip()
    if not url:
        return {"ok": False, "error": "缺少 url 参数"}
    max_chars = arguments.get("max_chars") or None
    page = await webfetch.fetch_url(url, max_chars=max_chars)
    if not page.ok:
        return {"ok": False, "url": url, "error": page.error}
    return {
        "ok": True,
        "url": page.final_url or page.url,
        "title": page.title,
        "text": page.text,
        "truncated": page.truncated,
        "status_code": page.status_code,
        "elapsed_ms": round(page.elapsed_ms, 1),
    }


async def _handle(request: dict) -> dict | None:
    method = request.get("method")
    msg_id = request.get("id")
    is_notification = "id" not in request

    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method in ("notifications/initialized", "initialized"):
        return None  # 通知，无响应
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name != "fetch":
            return _error(msg_id, -32602, f"未知工具：{name}")
        try:
            payload = await _call_fetch(arguments)
        except Exception as exc:  # 工具异常按 isError 结果回，不炸连接
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return _result(msg_id, {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
            "isError": not payload.get("ok", False),
        })
    if is_notification:
        return None  # 其余通知一律静默
    return _error(msg_id, -32601, f"方法不存在：{method}")


async def main() -> None:
    # stdio：一行一条 JSON-RPC。stderr 留给日志，绝不能污染 stdout。
    # 读用同步 readline 丢进线程池：Windows 上 asyncio 的 connect_read_pipe
    # 对 Proactor 循环有 WinError 6 句柄无效的坑，而 MCP 本来就是请求-响应
    # 串行协议，同步读行完全够用；写直接 stdout.write + flush。
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write(_error(None, -32700, "JSON 解析失败"))
            continue
        response = await _handle(request)
        if response is not None:
            _write(response)


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (BrokenPipeError, ConnectionResetError):
        pass
