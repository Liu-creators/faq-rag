"""基于 LangChain ChatOpenAI 的 OpenAI 兼容聊天客户端。"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from faq_rag.config import get_settings
from faq_rag.exceptions import LLMConfigError, LLMError


@lru_cache
def get_chat_model(*, temperature: float = 0.2, model: str | None = None) -> ChatOpenAI:
    """从环境变量配置构建可复用的 ChatOpenAI（OpenAI 兼容网关）。"""
    cfg = get_settings()
    if not cfg.llm_configured:
        raise LLMConfigError(
            "未设置 LLM_API_KEY。请在 .env 或进程环境变量中配置。"
        )
    return ChatOpenAI(
        model=model or cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=temperature,
    )


def get_llm_client() -> ChatOpenAI:
    """兼容旧名：返回默认温度的 ChatOpenAI。"""
    return get_chat_model()


def chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """调用聊天模型，返回助手文本内容。"""
    llm: BaseChatModel = get_chat_model(temperature=temperature, model=model)
    try:
        result = llm.invoke(messages)
    except LLMConfigError:
        raise
    except Exception as exc:  # 网络 / SDK / 上游错误
        raise LLMError(f"LLM 请求失败：{exc}") from exc

    content = getattr(result, "content", None)
    if content is None:
        raise LLMError("LLM 返回了空响应")
    text = content if isinstance(content, str) else str(content)
    if not text.strip():
        raise LLMError("LLM 返回了空响应")
    return text.strip()
