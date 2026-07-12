"""Similarity / vector index for FAQ question matching.

Owns corpus indexing, vectorization, and top-k search.
Swap ExactContainmentIndex for Word2Vec / BM25 / embedding indexes later —
FAQRetriever should only feed documents and map hits back to FAQ records.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence


def normalize_text(text: str) -> str:
    """Lightweight normalize shared by simple string indexes."""
    return "".join(text.casefold().split())


@dataclass(frozen=True)
class IndexedDocument:
    """One searchable text tied to an external id (e.g. FAQ id)."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class SimilarityHit:
    doc_id: str
    text: str
    score: float


class SimilarityIndex(Protocol):
    """Build an index over documents, then search without the caller scanning the corpus."""

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        """Replace index contents. Implementations may vectorize / build vocab here."""

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        """Return up to ``top_k`` unique ``doc_id`` hits, best score first."""


class ExactContainmentIndex:
    """Skeleton index: keeps texts in memory and scores with exact / containment.

    Scanning stays inside this module so FAQRetriever does not walk the corpus.
    Later Word2VecIndex would keep vectors/词典 here and ANN-search in ``search``.
    """

    def __init__(self) -> None:
        self._documents: list[IndexedDocument] = []

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        self._documents = [
            IndexedDocument(doc_id=doc.doc_id, text=doc.text)
            for doc in documents
            if doc.text.strip()
        ]

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        if top_k <= 0 or not query.strip() or not self._documents:
            return []

        best_by_doc: dict[str, SimilarityHit] = {}
        for doc in self._documents:
            score = self._score_pair(query, doc.text)
            if score <= 0:
                continue
            prev = best_by_doc.get(doc.doc_id)
            if prev is None or score > prev.score:
                best_by_doc[doc.doc_id] = SimilarityHit(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=score,
                )

        ranked = sorted(best_by_doc.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _score_pair(query: str, candidate: str) -> float:
        q = normalize_text(query)
        c = normalize_text(candidate)
        if not q or not c:
            return 0.0
        if q == c:
            return 1.0
        if q in c or c in q:
            shorter, longer = sorted((q, c), key=len)
            return min(0.85, 0.6 + 0.25 * (len(shorter) / len(longer)))
        return 0.0
