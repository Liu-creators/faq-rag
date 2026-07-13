"""基于统一 Settings 的 OpenAI 兼容聊天客户端。"""

from __future__ import annotations

from functools import lru_cache

from openai import APIError, OpenAI

from faq_rag.config import get_settings
from faq_rag.exceptions import LLMConfigError, LLMError


@lru_cache
def get_llm_client() -> OpenAI:
    """从环境变量配置构建可复用的 OpenAI 客户端。"""
    cfg = get_settings()
    if not cfg.llm_configured:
        raise LLMConfigError(
            "未设置 LLM_API_KEY。请在 .env 或进程环境变量中配置。"
        )
    return OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)


def chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """调用 chat.completions，返回助手文本内容。"""
    cfg = get_settings()
    client = get_llm_client()
    try:
        completion = client.chat.completions.create(
            model=model or cfg.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
    except APIError as exc:
        raise LLMError(f"LLM 请求失败：{exc}") from exc
    except Exception as exc:  # 网络 / SDK 意外错误
        raise LLMError(f"LLM 请求失败：{exc}") from exc

    choice = completion.choices[0] if completion.choices else None
    content = choice.message.content if choice and choice.message else None
    if not content or not str(content).strip():
        raise LLMError("LLM 返回了空响应")
    return str(content).strip()
