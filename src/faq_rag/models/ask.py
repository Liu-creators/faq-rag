"""Ask / answer pipeline schemas."""

from enum import Enum

from pydantic import BaseModel, Field

from faq_rag.models.faq import FAQ


class AnswerRoute(str, Enum):
    """Which branch produced the final answer."""

    FAQ_VERBATIM = "faq_verbatim"  # 极高置信：原样返回
    FAQ_REWRITE = "faq_rewrite"  # 高置信：基于 FAQ 改写
    DOC_RAG = "doc_rag"  # 未命中/偏低：文档 RAG 兜底


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户提问")


class FAQMatch(BaseModel):
    """Best FAQ hit from the first-stage retriever."""

    faq: FAQ
    score: float = Field(..., ge=0.0, le=1.0, description="占位置信度分数")
    matched_text: str = Field(..., description="实际命中的标准问或相似问")


class AskResponse(BaseModel):
    question: str
    answer: str
    route: AnswerRoute
    confidence: float | None = Field(
        default=None,
        description="FAQ 命中置信度；走文档 RAG 时可能为 None 或较低分",
    )
    faq_match: FAQMatch | None = None
    notes: str | None = Field(
        default=None,
        description="骨架阶段说明，便于联调时看清走了哪条分支",
    )
