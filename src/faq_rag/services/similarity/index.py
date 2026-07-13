"""FAQ 问题匹配用的相似度 / 向量索引。

负责语料建库、向量化与 top-k 检索。
优先顺序：Milvus 混合检索 → Word2VecIndex → ExactContainmentIndex。
FAQRetriever 只负责喂入文档并映射命中结果。
"""

from dataclasses import dataclass
from typing import Protocol, Sequence


def normalize_text(text: str) -> str:
    """简易字符串索引共用的轻量归一化。"""
    return "".join(text.casefold().split())


@dataclass(frozen=True)
class IndexedDocument:
    """一条可检索文本，绑定外部 id（如 FAQ id）。"""

    doc_id: str
    text: str


@dataclass(frozen=True)
class SimilarityHit:
    doc_id: str
    text: str
    score: float


class SimilarityIndex(Protocol):
    """对文档建索引，调用方无需自行扫描语料即可检索。"""

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        """替换索引内容。实现方可在此做向量化 / 建词表。"""

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        """返回最多 ``top_k`` 条唯一 ``doc_id`` 命中，按分数从高到低。"""


class ExactContainmentIndex:
    """骨架索引：文本存内存，用精确匹配 / 包含关系打分。

    扫描留在本模块内，避免 FAQRetriever 遍历语料。
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
