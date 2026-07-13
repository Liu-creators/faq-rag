"""FAQ 一阶段检索：将 FAQ 文本同步进 SimilarityIndex，再执行搜索。"""

from __future__ import annotations

from dataclasses import dataclass

from faq_rag.config import Settings, get_settings
from faq_rag.models.faq import FAQ
from faq_rag.services.faq.store import FAQStore, faq_store
from faq_rag.services.similarity import (
    ExactContainmentIndex,
    IndexedDocument,
    SimilarityIndex,
    Word2VecIndex,
)
from faq_rag.services.similarity.embedder import (
    DenseEmbedder,
    OpenAICompatibleEmbedder,
    Word2VecEmbedder,
)
from faq_rag.services.similarity.milvus_index import MilvusHybridIndex


@dataclass(frozen=True)
class ScoredFAQ:
    faq: FAQ
    score: float
    matched_text: str


def build_dense_embedder(settings: Settings | None = None) -> DenseEmbedder:
    """按配置选择稠密编码器：优先 OpenAI 兼容 API，否则 Word2Vec。"""
    cfg = settings or get_settings()
    if cfg.openai_embedding_configured:
        return OpenAICompatibleEmbedder(
            api_key=cfg.embedding_api_key or "no-key",
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
        )
    if cfg.word2vec_configured:
        return Word2VecEmbedder(cfg.word2vec_model_path)
    raise RuntimeError(
        "Milvus 混合检索需要稠密向量编码器：请配置 WORD2VEC_MODEL_PATH，"
        "或设置 EMBEDDING_MODEL + EMBEDDING_DIM（可选 EMBEDDING_BASE_URL / EMBEDDING_API_KEY）。"
    )


def _default_index() -> SimilarityIndex:
    settings = get_settings()
    if settings.milvus_configured:
        return MilvusHybridIndex(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection=settings.milvus_collection,
            embedder=build_dense_embedder(settings),
        )
    if settings.word2vec_configured:
        return Word2VecIndex(settings.word2vec_model_path)
    return ExactContainmentIndex()


class FAQRetriever:
    """可插拔 SimilarityIndex 之上的薄 FAQ 适配层。

    自身不做打分或候选扫描 — 向量化 / 匹配在 ``similarity`` 中完成
    （Milvus 混合 / Word2Vec / ExactContainment）。
    """

    def __init__(
        self,
        store: FAQStore | None = None,
        index: SimilarityIndex | None = None,
    ) -> None:
        self._store = store or faq_store
        # index=None 时惰性创建，避免 import 时强连 Milvus / 加载模型。
        self._index: SimilarityIndex | None = index

    def _get_index(self) -> SimilarityIndex:
        if self._index is None:
            self._index = _default_index()
        return self._index

    def retrieve(self, question: str, *, top_k: int = 1) -> list[ScoredFAQ]:
        if not question.strip() or top_k <= 0:
            return []

        self.sync_index()
        hits = self._get_index().search(question, top_k=top_k)

        results: list[ScoredFAQ] = []
        for hit in hits:
            faq = self._store.find(hit.doc_id)
            if faq is None or not faq.enabled:
                continue
            results.append(
                ScoredFAQ(faq=faq, score=hit.score, matched_text=hit.text)
            )
        return results

    def sync_index(self) -> None:
        """将已启用 FAQ 的标准问 / 相似问推入相似度索引。

        每次全量重建。后续可改为 CRUD 时增量 upsert。
        """
        documents: list[IndexedDocument] = []
        for faq in self._store.list(enabled=True):
            for text in (faq.question, *faq.similar_questions):
                if text.strip():
                    documents.append(IndexedDocument(doc_id=faq.id, text=text))
        self._get_index().build(documents)


faq_retriever = FAQRetriever()
