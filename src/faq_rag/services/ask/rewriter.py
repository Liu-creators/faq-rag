"""中置信区间：基于 FAQ 锚定的答案改写（LangChain LCEL）。"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from faq_rag.exceptions import LLMError
from faq_rag.models.faq import FAQ
from faq_rag.services.llm import get_chat_model

_SYSTEM_PROMPT = """\
你是客服问答改写助手。你必须基于给定的 FAQ 标准答案进行改写，遵守：
1. 不得新增、删改或推测 FAQ 中没有的事实、数字、步骤或承诺。
2. 可调整措辞与语气，使回答更贴合用户当前提问的表达。
3. 直接输出改写后的答案正文，不要加前言、标题或解释。
"""

_USER_TEMPLATE = """\
用户提问：{question}

FAQ 标准问：{faq_question}
FAQ 标准答：{faq_answer}

请在不改变事实的前提下，改写标准答以回应用户提问。
"""


@lru_cache
def get_rewrite_chain() -> Runnable:
    """FAQ 锚定改写链：prompt | ChatOpenAI | StrOutputParser。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_TEMPLATE),
        ]
    )
    return prompt | get_chat_model(temperature=0.2) | StrOutputParser()


def rewrite_from_faq(*, question: str, faq: FAQ) -> str:
    """在锁定事实的前提下，按用户提问改写 FAQ 标准答。"""
    chain = get_rewrite_chain()
    try:
        text = chain.invoke(
            {
                "question": question,
                "faq_question": faq.question,
                "faq_answer": faq.answer,
            },
            config={"run_name": "faq_rewrite"},
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"LLM 请求失败：{exc}") from exc

    if not str(text).strip():
        raise LLMError("LLM 返回了空响应")
    return str(text).strip()


def rewrite_from_faq_stream(*, question: str, faq: FAQ):
    """流式返回改写结果。"""
    chain = get_rewrite_chain()
    try:
        yield from chain.stream(
            {
                "question": question,
                "faq_question": faq.question,
                "faq_answer": faq.answer,
            },
            config={"run_name": "faq_rewrite_stream"},
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"LLM 请求失败：{exc}") from exc
