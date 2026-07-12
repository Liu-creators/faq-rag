"""Answer rewrite stub (LLM rewrite locked to FAQ facts — not implemented yet)."""

from faq_rag.models.faq import FAQ


def rewrite_from_faq(*, question: str, faq: FAQ) -> str:
    """Skeleton: mark rewrite path without calling an LLM."""
    return (
        f"[rewrite-placeholder] 针对提问「{question}」，"
        f"基于 FAQ「{faq.question}」的标准答：{faq.answer}"
    )
