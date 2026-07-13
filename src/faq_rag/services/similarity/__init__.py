"""用于问题匹配的相似度 / 向量索引。"""

from faq_rag.services.similarity.index import (
    ExactContainmentIndex,
    IndexedDocument,
    SimilarityHit,
    SimilarityIndex,
    normalize_text,
)
from faq_rag.services.similarity.milvus_index import MilvusHybridIndex
from faq_rag.services.similarity.word2vec import Word2VecIndex

__all__ = [
    "ExactContainmentIndex",
    "IndexedDocument",
    "MilvusHybridIndex",
    "SimilarityHit",
    "SimilarityIndex",
    "Word2VecIndex",
    "normalize_text",
]
