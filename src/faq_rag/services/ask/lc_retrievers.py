"""将现有 FAQ / 文档检索适配为 LangChain BaseRetriever（不改动 Milvus 实现）。"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from faq_rag.config import get_settings
from faq_rag.models.document import DocChunk
from faq_rag.models.document import Document as DocRecord
from faq_rag.models.faq import FAQ
from faq_rag.services.doc.retriever import DocChunkRetriever, ScoredChunk, doc_chunk_retriever
from faq_rag.services.faq.retriever import FAQRetriever, ScoredFAQ, faq_retriever


class FAQLangChainRetriever(BaseRetriever):
    """包装 ``FAQRetriever``，供 LangSmith 追踪检索 span。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: FAQRetriever = Field(default_factory=lambda: faq_retriever)
    top_k: int = 1

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager  # 检索侧无需额外 callback
        hits = self.inner.retrieve(query, top_k=self.top_k)
        return [_faq_hit_to_document(hit) for hit in hits]


class DocChunkLangChainRetriever(BaseRetriever):
    """包装 ``DocChunkRetriever``，供 LangSmith 追踪检索 span。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: DocChunkRetriever = Field(default_factory=lambda: doc_chunk_retriever)
    top_k: int | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        settings = get_settings()
        limit = self.top_k if self.top_k is not None else settings.doc_rag_top_k
        hits = self.inner.retrieve(query, top_k=limit)
        return [_chunk_hit_to_document(hit) for hit in hits]


def _faq_hit_to_document(hit: ScoredFAQ) -> Document:
    return Document(
        page_content=hit.matched_text,
        metadata={
            "score": hit.score,
            "faq_id": hit.faq.id,
            "matched_text": hit.matched_text,
            "faq": hit.faq.model_dump(mode="json"),
            "kind": "faq",
        },
    )


def _chunk_hit_to_document(hit: ScoredChunk) -> Document:
    heading = hit.chunk.heading or ""
    content = hit.chunk.content
    page = f"{heading}\n\n{content}".strip() if heading else content
    return Document(
        page_content=page,
        metadata={
            "score": hit.score,
            "chunk_id": hit.chunk.id,
            "document_id": hit.document.id,
            "title": hit.document.title,
            "heading": hit.chunk.heading,
            "source_path": hit.document.source_path,
            "chunk": hit.chunk.model_dump(mode="json"),
            "document": hit.document.model_dump(mode="json"),
            "kind": "doc_chunk",
        },
    )


def documents_to_scored_faqs(docs: list[Document]) -> list[ScoredFAQ]:
    """将 LangChain Document 还原为领域 ScoredFAQ（优先用 metadata 快照）。"""
    results: list[ScoredFAQ] = []
    for doc in docs:
        meta: dict[str, Any] = dict(doc.metadata or {})
        faq_raw = meta.get("faq")
        if not isinstance(faq_raw, dict):
            continue
        faq = FAQ.model_validate(faq_raw)
        if not faq.enabled:
            continue
        score = float(meta.get("score", 0.0))
        matched = str(meta.get("matched_text") or doc.page_content)
        results.append(ScoredFAQ(faq=faq, score=score, matched_text=matched))
    return results


def documents_to_scored_chunks(docs: list[Document]) -> list[ScoredChunk]:
    """将 LangChain Document 还原为领域 ScoredChunk（优先用 metadata 快照）。"""
    results: list[ScoredChunk] = []
    for doc in docs:
        meta: dict[str, Any] = dict(doc.metadata or {})
        chunk_raw = meta.get("chunk")
        document_raw = meta.get("document")
        if not isinstance(chunk_raw, dict) or not isinstance(document_raw, dict):
            continue
        chunk = DocChunk.model_validate(chunk_raw)
        document = DocRecord.model_validate(document_raw)
        if not document.enabled:
            continue
        score = float(meta.get("score", 0.0))
        results.append(ScoredChunk(chunk=chunk, document=document, score=score))
    return results
