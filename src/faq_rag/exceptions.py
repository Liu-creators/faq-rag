"""FAQ RAG 应用的领域异常。

在服务层抛出；API 层通过 ``faq_rag.api.exception_handlers`` 映射为 HTTP 响应。
"""


class AppError(Exception):
    """应用错误基类。子类设置 ``status_code`` / ``code``。"""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.message = message
        if code is not None:
            self.code = code
        super().__init__(message)


class NotFoundError(AppError):
    """请求的资源不存在。"""

    status_code = 404
    code = "not_found"


class FAQNotFoundError(NotFoundError):
    """FAQ id 在存储中不存在。"""

    code = "faq_not_found"

    def __init__(self, faq_id: str) -> None:
        self.faq_id = faq_id
        super().__init__(f"未找到 FAQ：{faq_id}")


class LLMConfigError(AppError):
    """LLM 凭证 / base URL 缺失或无效。"""

    status_code = 503
    code = "llm_config_error"


class LLMError(AppError):
    """上游 LLM 调用失败或返回不可用内容。"""

    status_code = 502
    code = "llm_error"
