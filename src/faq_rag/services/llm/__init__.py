"""共享 LLM 客户端（OpenAI 兼容）。"""

from faq_rag.services.llm.client import chat_completion, get_llm_client

__all__ = [
    "chat_completion",
    "get_llm_client",
]
