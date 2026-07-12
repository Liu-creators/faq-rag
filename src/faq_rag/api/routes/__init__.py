"""API route aggregation."""

from fastapi import APIRouter

from . import ask, faqs, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(faqs.router)
api_router.include_router(ask.router)
