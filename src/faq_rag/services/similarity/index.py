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

    def upsert(self, documents: Sequence[IndexedDocument]) -> None:
        """按条写入或覆盖；不自动清理同 ``doc_id`` 下已变更的旧文本行。"""

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除该 ``doc_id`` 下全部索引行。"""

    def search(self, query: str, *, top_k: int = 1) -> list[SimilarityHit]:
        """返回最多 ``top_k`` 条唯一 ``doc_id`` 命中，按分数从高到低。"""

    def has_data(self) -> bool:
        """检查索引是否已有数据（用于跳过重启后的全量冷启动建库）。"""


class ExactContainmentIndex:
    """骨架索引：文本存内存，用精确匹配 / 包含关系打分。

    扫描留在本模块内，避免 FAQRetriever 遍历语料。
    """

    def __init__(self) -> None:
        self._documents: list[IndexedDocument] = []

    def build(self, documents: Sequence[IndexedDocument]) -> None:
        self._documents = [
            IndexedDocument(doc_id=doc.doc_id, text=doc.text.strip())
            for doc in documents
            if doc.text.strip()
        ]

    def upsert(self, documents: Sequence[IndexedDocument]) -> None:
        incoming = [
            IndexedDocument(doc_id=doc.doc_id, text=doc.text.strip())
            for doc in documents
            if doc.text.strip()
        ]
        if not incoming:
            return
        keys = {(doc.doc_id, doc.text) for doc in incoming}
        self._documents = [
            doc for doc in self._documents if (doc.doc_id, doc.text) not in keys
        ]
        self._documents.extend(incoming)

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._documents = [doc for doc in self._documents if doc.doc_id != doc_id]

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

    def has_data(self) -> bool:
        return bool(self._documents)

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
