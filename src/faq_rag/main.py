"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from faq_rag import __version__
from faq_rag.api.exception_handlers import register_exception_handlers
from faq_rag.api.routes import api_router
from faq_rag.db import init_db

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FAQ RAG",
    description="FAQ 检索增强生成 API",
    version=__version__,
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(api_router)


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    return {"message": "FAQ RAG API 运行中", "docs": "/docs"}


def run() -> None:
    """启动开发服务器。"""
    import uvicorn

    uvicorn.run(
        "faq_rag.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
