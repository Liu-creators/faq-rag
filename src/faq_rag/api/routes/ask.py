"""提问入口 — 用户问题 → FAQ 闸门 → 文档 RAG 兜底。"""

from fastapi import APIRouter

from faq_rag.models.ask import AskRequest, AskResponse
from faq_rag.services.ask import ask_pipeline

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """提问入口：先 FAQ，再按置信度分流（中置信 FAQ 改写；低置信 / 未命中走文档 RAG）。"""
    return ask_pipeline.run(request)
