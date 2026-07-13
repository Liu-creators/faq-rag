"""API 路由聚合。"""

from fastapi import APIRouter

from . import ask, docs, faqs, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(faqs.router)
api_router.include_router(docs.router)
api_router.include_router(ask.router)
