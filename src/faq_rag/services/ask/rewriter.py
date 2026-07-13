"""中置信区间：基于 FAQ 锚定的答案改写。"""

from __future__ import annotations

from faq_rag.models.faq import FAQ
from faq_rag.services.llm import chat_completion

_SYSTEM_PROMPT = """\
你是客服问答改写助手。你必须基于给定的 FAQ 标准答案进行改写，遵守：
1. 不得新增、删改或推测 FAQ 中没有的事实、数字、步骤或承诺。
2. 可调整措辞与语气，使回答更贴合用户当前提问的表达。
3. 直接输出改写后的答案正文，不要加前言、标题或解释。
"""


def rewrite_from_faq(*, question: str, faq: FAQ) -> str:
    """在锁定事实的前提下，按用户提问改写 FAQ 标准答。"""
    user_prompt = (
        f"用户提问：{question}\n\n"
        f"FAQ 标准问：{faq.question}\n"
        f"FAQ 标准答：{faq.answer}\n\n"
        "请在不改变事实的前提下，改写标准答以回应用户提问。"
    )
    return chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
