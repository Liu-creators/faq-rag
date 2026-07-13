"""Word2Vec 相似度索引：预训练 KeyedVectors + jieba + OOV 哈希向量。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from faq_rag.services.similarity.embedder import Word2VecEmbedder, cosine_similarity
from faq_rag.services.similarity.index import IndexedDocument, SimilarityHit


class Word2VecIndex:
    """基于预训练 gensim KeyedVectors 文件（``.kv``）的 FAQ 问题索引。

    句向量为词表内词向量与哈希 OOV 伪向量的均值，再做 L2 归一化。
    检索用余弦相似度（单位向量点积），线性扫描。
    """

    def __init__(self, model_path: str | Path) -> None:
        self._embedder = Word2VecEmbedder(model_path)
        self._entries: list[tuple[str, str, list[float]]] = []

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        docs = [
            IndexedDocument(doc_id=doc.doc_id, text=doc.text.strip())
            for doc in documents
            if doc.text.strip()
        ]
        if not docs:
            self._entries = []
            return
        vectors = self._embedder.embed([doc.text for doc in docs])
        self._entries = [
            (doc.doc_id, doc.text, vec)
            for doc, vec in zip(docs, vectors, strict=True)
            if any(v != 0.0 for v in vec)
        ]

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        if top_k <= 0 or not query.strip() or not self._entries:
            return []

        query_vec = self._embedder.embed([query])[0]
        if all(v == 0.0 for v in query_vec):
            return []

        best_by_doc: dict[str, SimilarityHit] = {}
        for doc_id, text, vector in self._entries:
            score = cosine_similarity(query_vec, vector)
            if score <= 0:
                continue
            prev = best_by_doc.get(doc_id)
            if prev is None or score > prev.score:
                best_by_doc[doc_id] = SimilarityHit(
                    doc_id=doc_id,
                    text=text,
                    score=score,
                )

        ranked = sorted(best_by_doc.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]
