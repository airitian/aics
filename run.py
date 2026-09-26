"""开发启动脚本：python run.py

生产建议：
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
（注意：>1 worker 时内置限流器是进程内的，需换成 Redis 实现，见 app/ratelimit.py 顶部说明）
"""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "1") == "1"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
