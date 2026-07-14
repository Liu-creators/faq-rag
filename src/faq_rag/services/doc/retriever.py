"""文档切块检索：将 chunk 同步进 SimilarityIndex，再执行搜索。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from faq_rag.config import Settings, get_settings
from faq_rag.models.document import DocChunk, Document
from faq_rag.services.doc.store import DocumentStore, document_store
from faq_rag.services.faq.retriever import build_dense_embedder
from faq_rag.services.similarity import (
    ExactContainmentIndex,
    IndexedDocument,
    SimilarityIndex,
    Word2VecIndex,
)
from faq_rag.services.similarity.milvus_index import MilvusHybridIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocChunk
    document: Document
    score: float


def _default_doc_index() -> SimilarityIndex:
    settings = get_settings()
    if settings.milvus_configured:
        return MilvusHybridIndex(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection=settings.milvus_doc_collection,
            embedder=build_dense_embedder(settings),
        )
    if settings.word2vec_configured:
        return Word2VecIndex(settings.word2vec_model_path)
    return ExactContainmentIndex()


def _index_text(chunk: DocChunk) -> str:
    """索引文本：标题路径 + 正文，利于 BM25 命中章节名。"""
    if chunk.heading:
        return f"{chunk.heading}\n\n{chunk.content}"
    return chunk.content


def _chunk_document(chunk: DocChunk) -> IndexedDocument:
    return IndexedDocument(doc_id=chunk.id, text=_index_text(chunk))


class DocChunkRetriever:
    """可插拔 SimilarityIndex 之上的文档切块适配层。

    索引由 ingest / reindex 维护；首次检索前若尚未就绪则全量引导一次。
    """

    def __init__(
        self,
        store: DocumentStore | None = None,
        index: SimilarityIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._store = store or document_store
        self._index: SimilarityIndex | None = index
        self._settings = settings
        self._ready = False

    def _get_index(self) -> SimilarityIndex:
        if self._index is None:
            self._index = _default_doc_index()
        return self._index

    def _ensure_ready(self) -> None:
        if not self._ready:
            if self._get_index().has_data():
                self._ready = True
                logger.info("DocChunkRetriever 检测到索引已有数据，跳过全量启动")
            else:
                self.sync_index()

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[ScoredChunk]:
        settings = self._settings or get_settings()
        limit = top_k if top_k is not None else settings.doc_rag_top_k
        if not question.strip() or limit <= 0:
            return []

        self._ensure_ready()
        hits = self._get_index().search(question, top_k=limit)

        results: list[ScoredChunk] = []
        for hit in hits:
            chunk = self._store.find_chunk(hit.doc_id)
            if chunk is None:
                continue
            doc = self._store.find(chunk.document_id)
            if doc is None or not doc.enabled:
                continue
            results.append(ScoredChunk(chunk=chunk, document=doc, score=hit.score))
        return results

    def sync_index(self) -> int:
        """将已启用文档的全部切块全量推入相似度索引。返回索引条数。"""
        documents: list[IndexedDocument] = []
        for chunk, _doc in self._store.list_chunks_with_documents(enabled_docs_only=True):
            text = _index_text(chunk).strip()
            if text:
                documents.append(IndexedDocument(doc_id=chunk.id, text=text))
        self._get_index().build(documents)
        self._ready = True
        return len(documents)

    def upsert_chunks(self, chunks: list[DocChunk]) -> None:
        """增量写入切块；索引未就绪时跳过。"""
        if not self._ready or not chunks:
            return
        try:
            docs = [_chunk_document(c) for c in chunks if c.content.strip()]
            if docs:
                self._get_index().upsert(docs)
        except Exception:
            logger.exception("文档切块增量索引失败，将在下次检索时全量重建")
            self._ready = False

    def remove_chunk(self, chunk_id: str) -> None:
        if not self._ready:
            return
        try:
            self._get_index().delete_by_doc_id(chunk_id)
        except Exception:
            logger.exception(
                "文档切块 %s 索引删除失败，将在下次检索时全量重建", chunk_id
            )
            self._ready = False

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.remove_chunk(chunk_id)


doc_chunk_retriever = DocChunkRetriever()
