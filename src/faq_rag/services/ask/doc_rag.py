"""文档 RAG：粗粒度 / 长尾兜底（检索 + 生成）。"""

from __future__ import annotations

from dataclasses import dataclass

from faq_rag.config import get_settings
from faq_rag.models.ask import DocSource
from faq_rag.services.ask.generator import generate_from_chunks
from faq_rag.services.doc.retriever import DocChunkRetriever, doc_chunk_retriever


@dataclass(frozen=True)
class DocRagResult:
    answer: str
    sources: list[DocSource]
    notes: str


def answer_from_documents(
    *,
    question: str,
    retriever: DocChunkRetriever | None = None,
    top_k: int | None = None,
) -> DocRagResult:
    """检索文档切块并用 LLM 生成带引用的回答。"""
    index = retriever or doc_chunk_retriever
    settings = get_settings()
    limit = top_k if top_k is not None else settings.doc_rag_top_k
    hits = index.retrieve(question, top_k=limit)

    if not hits:
        return DocRagResult(
            answer="知识库中未找到与该问题相关的文档内容，暂时无法回答。",
            sources=[],
            notes="文档 RAG：无检索命中",
        )

    answer = generate_from_chunks(question=question, hits=hits)
    sources = [
        DocSource(
            document_id=hit.document.id,
            chunk_id=hit.chunk.id,
            title=hit.document.title,
            heading=hit.chunk.heading,
            score=hit.score,
            source_path=hit.document.source_path,
        )
        for hit in hits
    ]
    return DocRagResult(
        answer=answer,
        sources=sources,
        notes=f"文档 RAG：召回 {len(sources)} 个切块后生成",
    )
