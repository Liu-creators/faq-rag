"""FAQ first-stage retrieval: sync FAQ texts into a SimilarityIndex, then search."""

from dataclasses import dataclass

from faq_rag.models.faq import FAQ
from faq_rag.services.faq.store import FAQStore, faq_store
from faq_rag.services.similarity import (
    ExactContainmentIndex,
    IndexedDocument,
    SimilarityIndex,
)


@dataclass(frozen=True)
class ScoredFAQ:
    faq: FAQ
    score: float
    matched_text: str


class FAQRetriever:
    """Thin FAQ adapter over a pluggable SimilarityIndex.

    Does not score or scan candidates itself — vectorization / matching live in
    ``similarity`` (ExactContainment now; Word2Vec / BM25 later).
    """

    def __init__(
        self,
        store: FAQStore | None = None,
        index: SimilarityIndex | None = None,
    ) -> None:
        self._store = store or faq_store
        # Each retriever gets its own index unless one is injected (e.g. Word2VecIndex).
        self._index = index if index is not None else ExactContainmentIndex()

    def retrieve(self, question: str, *, top_k: int = 1) -> list[ScoredFAQ]:
        if not question.strip() or top_k <= 0:
            return []

        self.sync_index()
        hits = self._index.search(question, top_k=top_k)

        results: list[ScoredFAQ] = []
        for hit in hits:
            faq = self._store.get(hit.doc_id)
            if faq is None or not faq.enabled:
                continue
            results.append(
                ScoredFAQ(faq=faq, score=hit.score, matched_text=hit.text)
            )
        return results

    def sync_index(self) -> None:
        """Push enabled FAQ standard / similar questions into the similarity index.

        Skeleton: full rebuild each call. Later: incremental upsert on CRUD, or
        Word2Vec re-fit / reload 词典 inside the index implementation.
        """
        documents: list[IndexedDocument] = []
        for faq in self._store.list(enabled=True):
            for text in (faq.question, *faq.similar_questions):
                if text.strip():
                    documents.append(IndexedDocument(doc_id=faq.id, text=text))
        self._index.build(documents)


faq_retriever = FAQRetriever()
