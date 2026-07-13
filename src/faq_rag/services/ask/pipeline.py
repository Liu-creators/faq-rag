"""提问流水线：先 FAQ，再按置信度分流改写 / 文档 RAG。"""

from faq_rag.models.ask import AnswerRoute, AskRequest, AskResponse, FAQMatch
from faq_rag.services.ask.doc_rag import answer_from_documents
from faq_rag.services.ask.rewriter import rewrite_from_faq
from faq_rag.services.faq import FAQRetriever, faq_retriever

# 占位阈值 — 待真实检索分数就绪后再调优。
CONFIDENCE_VERBATIM = 0.95  # 极高：原样返回
CONFIDENCE_REWRITE = 0.90  # 高：允许改写


class AskPipeline:
    def __init__(self, retriever: FAQRetriever | None = None) -> None:
        self._retriever = retriever or faq_retriever

    def run(self, request: AskRequest) -> AskResponse:
        hits = self._retriever.retrieve(request.question, top_k=1)
        if not hits:
            rag = answer_from_documents(question=request.question)
            return AskResponse(
                question=request.question,
                answer=rag.answer,
                route=AnswerRoute.DOC_RAG,
                confidence=None,
                faq_match=None,
                sources=rag.sources or None,
                notes=rag.notes,
            )

        best = hits[0]
        match = FAQMatch(faq=best.faq, score=best.score, matched_text=best.matched_text)

        if best.score >= CONFIDENCE_VERBATIM:
            return AskResponse(
                question=request.question,
                answer=best.faq.answer,
                route=AnswerRoute.FAQ_VERBATIM,
                confidence=best.score,
                faq_match=match,
                sources=None,
                notes="置信度极高 → 原样返回 FAQ 标准答",
            )

        if best.score >= CONFIDENCE_REWRITE:
            return AskResponse(
                question=request.question,
                answer=rewrite_from_faq(question=request.question, faq=best.faq),
                route=AnswerRoute.FAQ_REWRITE,
                confidence=best.score,
                faq_match=match,
                sources=None,
                notes="置信度高 → FAQ 锚定 LLM 改写",
            )

        rag = answer_from_documents(question=request.question)
        return AskResponse(
            question=request.question,
            answer=rag.answer,
            route=AnswerRoute.DOC_RAG,
            confidence=best.score,
            faq_match=match,
            sources=rag.sources or None,
            notes=rag.notes,
        )


ask_pipeline = AskPipeline()
