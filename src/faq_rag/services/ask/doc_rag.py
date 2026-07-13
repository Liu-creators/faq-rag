"""文档 RAG 占位（粗粒度 / 长尾兜底 — 尚未实现）。"""


def answer_from_documents(*, question: str) -> str:
    """骨架：标记走文档 RAG 路径，暂不做检索与生成。"""
    return (
        f"[doc-rag-placeholder] 暂未命中高置信 FAQ，"
        f"将针对「{question}」走文档检索增强生成（待实现）。"
    )
