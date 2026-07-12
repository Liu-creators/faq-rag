"""Health check endpoints."""

from fastapi import APIRouter

from faq_rag import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
