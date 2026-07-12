"""Document RAG stub (coarse / long-tail fallback — not implemented yet)."""


def answer_from_documents(*, question: str) -> str:
    """Skeleton: mark doc-RAG path without retrieval or generation."""
    return (
        f"[doc-rag-placeholder] 暂未命中高置信 FAQ，"
        f"将针对「{question}」走文档检索增强生成（待实现）。"
    )
