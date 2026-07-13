"""文档 RAG 生成：基于检索片段的事实锁定回答（LangChain LCEL）。"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from faq_rag.exceptions import LLMError
from faq_rag.services.doc.retriever import ScoredChunk
from faq_rag.services.llm import get_chat_model

_SYSTEM_PROMPT = """\
你是知识库问答助手。你必须严格基于给定的检索片段回答用户问题，遵守：
1. 不得编造片段中没有的事实、数字、步骤或承诺。
2. 若片段不足以回答，明确说明知识库未覆盖或信息不足，不要猜测。
3. 用中文直接输出答案正文；可在相关处用 [n] 标注依据片段编号。
4. 不要输出与问题无关的前言或标题。
"""

_USER_TEMPLATE = """\
用户提问：{question}

检索片段：
{context}

请仅依据上述片段回答用户提问；无法回答时请明确说明。
"""


@lru_cache
def get_generate_chain() -> Runnable:
    """文档 RAG 生成链：prompt | ChatOpenAI | StrOutputParser。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_TEMPLATE),
        ]
    )
    return prompt | get_chat_model(temperature=0.2) | StrOutputParser()


def _format_context(hits: list[ScoredChunk]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        heading = hit.chunk.heading or "（无标题）"
        parts.append(
            f"[{i}] 文档：{hit.document.title}\n"
            f"章节：{heading}\n"
            f"内容：{hit.chunk.content}"
        )
    return "\n\n".join(parts)


def generate_from_chunks(*, question: str, hits: list[ScoredChunk]) -> str:
    """根据检索到的切块生成回答。"""
    if not hits:
        return "知识库中未找到与该问题相关的文档内容，暂时无法回答。"

    chain = get_generate_chain()
    try:
        text = chain.invoke(
            {
                "question": question,
                "context": _format_context(hits),
            },
            config={"run_name": "doc_rag_generate"},
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"LLM 请求失败：{exc}") from exc

    if not str(text).strip():
        raise LLMError("LLM 返回了空响应")
    return str(text).strip()
