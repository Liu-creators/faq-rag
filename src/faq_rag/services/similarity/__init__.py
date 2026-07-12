"""Similarity / vector indexes for question matching."""

from faq_rag.services.similarity.index import (
    ExactContainmentIndex,
    IndexedDocument,
    SimilarityHit,
    SimilarityIndex,
    normalize_text,
)

__all__ = [
    "ExactContainmentIndex",
    "IndexedDocument",
    "SimilarityHit",
    "SimilarityIndex",
    "normalize_text",
]
