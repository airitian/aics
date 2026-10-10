"""路由聚合。"""
from fastapi import APIRouter

from app.routers import (
    auth,
    chat,
    employees,
    insights,
    knowledge,
    platform,
    playground,
    sessions,
    testchat,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(platform.router)
api_router.include_router(employees.router)
api_router.include_router(knowledge.router)
api_router.include_router(chat.router)
api_router.include_router(sessions.router)
api_router.include_router(playground.router)
api_router.include_router(testchat.router)
api_router.include_router(insights.router)

__all__ = ["api_router"]
