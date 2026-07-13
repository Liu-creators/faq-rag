"""文档 RAG 生成：基于检索片段的事实锁定回答。"""

from __future__ import annotations

from faq_rag.services.doc.retriever import ScoredChunk
from faq_rag.services.llm import chat_completion

_SYSTEM_PROMPT = """\
你是知识库问答助手。你必须严格基于给定的检索片段回答用户问题，遵守：
1. 不得编造片段中没有的事实、数字、步骤或承诺。
2. 若片段不足以回答，明确说明知识库未覆盖或信息不足，不要猜测。
3. 用中文直接输出答案正文；可在相关处用 [n] 标注依据片段编号。
4. 不要输出与问题无关的前言或标题。
"""


def generate_from_chunks(*, question: str, hits: list[ScoredChunk]) -> str:
    """根据检索到的切块生成回答。"""
    if not hits:
        return "知识库中未找到与该问题相关的文档内容，暂时无法回答。"

    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        heading = hit.chunk.heading or "（无标题）"
        parts.append(
            f"[{i}] 文档：{hit.document.title}\n"
            f"章节：{heading}\n"
            f"内容：{hit.chunk.content}"
        )
    context = "\n\n".join(parts)
    user_prompt = (
        f"用户提问：{question}\n\n"
        f"检索片段：\n{context}\n\n"
        "请仅依据上述片段回答用户提问；无法回答时请明确说明。"
    )
    return chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
