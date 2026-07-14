"""文档 RAG：粗粒度 / 长尾兜底（LangChain 检索包装 + 生成）。"""

from __future__ import annotations

from dataclasses import dataclass

from faq_rag.config import get_settings
from faq_rag.models.ask import DocSource
from faq_rag.services.ask.generator import generate_from_chunks
from faq_rag.services.ask.lc_retrievers import (
    DocChunkLangChainRetriever,
    documents_to_scored_chunks,
)
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
    settings = get_settings()
    limit = top_k if top_k is not None else settings.doc_rag_top_k
    inner = retriever or doc_chunk_retriever
    lc_retriever = DocChunkLangChainRetriever(inner=inner, top_k=limit)
    docs = lc_retriever.invoke(question, config={"run_name": "doc_retrieve"})
    hits = documents_to_scored_chunks(docs)

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


def answer_from_documents_stream(
    *,
    question: str,
    retriever: DocChunkRetriever | None = None,
    top_k: int | None = None,
):
    """流式检索并生成回答，产出 metadata dict 和 chunk 字符串。"""
    settings = get_settings()
    limit = top_k if top_k is not None else settings.doc_rag_top_k
    inner = retriever or doc_chunk_retriever
    lc_retriever = DocChunkLangChainRetriever(inner=inner, top_k=limit)
    docs = lc_retriever.invoke(question, config={"run_name": "doc_retrieve"})
    hits = documents_to_scored_chunks(docs)

    if not hits:
        yield {
            "type": "meta",
            "sources": [],
            "notes": "文档 RAG：无检索命中",
        }
        yield "知识库中未找到与该问题相关的文档内容，暂时无法回答。"
        return

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
    yield {
        "type": "meta",
        "sources": [s.model_dump() for s in sources],
        "notes": f"文档 RAG：召回 {len(sources)} 个切块后生成",
    }

    from faq_rag.services.ask.generator import get_generate_chain, _format_context
    chain = get_generate_chain()
    try:
        yield from chain.stream(
            {
                "question": question,
                "context": _format_context(hits),
            },
            config={"run_name": "doc_rag_generate_stream"},
        )
    except Exception as exc:
        from faq_rag.exceptions import LLMError
        raise LLMError(f"LLM 请求失败：{exc}") from exc
