"""将领域异常映射为 HTTP JSON 响应。"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from faq_rag.exceptions import AppError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )
