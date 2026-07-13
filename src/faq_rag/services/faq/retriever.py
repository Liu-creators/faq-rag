"""FAQ 一阶段检索：将 FAQ 文本同步进 SimilarityIndex，再执行搜索。"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# 变更这些字段时需要同步相似度索引。
_INDEX_FIELDS = frozenset({"question", "similar_questions", "enabled"})


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


def _faq_documents(faq: FAQ) -> list[IndexedDocument]:
    documents: list[IndexedDocument] = []
    for text in (faq.question, *faq.similar_questions):
        if text.strip():
            documents.append(IndexedDocument(doc_id=faq.id, text=text))
    return documents


class FAQRetriever:
    """可插拔 SimilarityIndex 之上的薄 FAQ 适配层。

    自身不做打分或候选扫描 — 向量化 / 匹配在 ``similarity`` 中完成
    （Milvus 混合 / Word2Vec / ExactContainment）。

    索引由 CRUD 增量维护；首次检索前若尚未就绪则全量引导一次。
    """

    def __init__(
        self,
        store: FAQStore | None = None,
        index: SimilarityIndex | None = None,
    ) -> None:
        self._store = store or faq_store
        # index=None 时惰性创建，避免 import 时强连 Milvus / 加载模型。
        self._index: SimilarityIndex | None = index
        self._ready = False

    def _get_index(self) -> SimilarityIndex:
        if self._index is None:
            self._index = _default_index()
        return self._index

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.sync_index()

    def retrieve(self, question: str, *, top_k: int = 1) -> list[ScoredFAQ]:
        if not question.strip() or top_k <= 0:
            return []

        self._ensure_ready()
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
        """将已启用 FAQ 的标准问 / 相似问全量推入相似度索引（启动 / 修复）。"""
        documents: list[IndexedDocument] = []
        for faq in self._store.list(enabled=True):
            documents.extend(_faq_documents(faq))
        self._get_index().build(documents)
        self._ready = True

    def upsert_faq(self, faq: FAQ) -> None:
        """按单条 FAQ 增量同步索引；禁用则删除。

        若索引尚未引导，跳过（下次检索会全量同步，已包含该 FAQ）。
        """
        if not self._ready:
            return
        try:
            index = self._get_index()
            index.delete_by_doc_id(faq.id)
            if not faq.enabled:
                return
            documents = _faq_documents(faq)
            if documents:
                index.upsert(documents)
        except Exception:
            logger.exception(
                "FAQ %s 增量索引失败，将在下次检索时全量重建", faq.id
            )
            self._ready = False

    def remove_faq(self, faq_id: str) -> None:
        """从索引中移除该 FAQ；索引未就绪时跳过。"""
        if not self._ready:
            return
        try:
            self._get_index().delete_by_doc_id(faq_id)
        except Exception:
            logger.exception(
                "FAQ %s 索引删除失败，将在下次检索时全量重建", faq_id
            )
            self._ready = False

    @staticmethod
    def index_fields_changed(patch: dict) -> bool:
        """判断 FAQUpdate 的 patch 是否影响索引。"""
        return bool(_INDEX_FIELDS & patch.keys())


faq_retriever = FAQRetriever()
