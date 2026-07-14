"""共享 LLM 客户端（LangChain ChatOpenAI / OpenAI 兼容）。"""

from faq_rag.services.llm.client import chat_completion, get_chat_model, get_llm_client

__all__ = [
    "chat_completion",
    "get_chat_model",
    "get_llm_client",
]
