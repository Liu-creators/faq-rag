"""FastAPI 应用入口。"""

import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from faq_rag import __version__
from faq_rag.api.exception_handlers import register_exception_handlers
from faq_rag.api.routes import api_router
from faq_rag.config import apply_langsmith_env, get_settings
from faq_rag.db import init_db

load_dotenv()
# 尽早同步 LangSmith 环境变量，便于后续 LangChain 调用上报 trace
apply_langsmith_env(get_settings())

logger = logging.getLogger(__name__)


async def warmup_indexes() -> None:
    """后台异步预热向量数据库连接与模型加载（模拟一次查询建立长链接）。"""
    try:
        from faq_rag.services.doc.retriever import doc_chunk_retriever
        from faq_rag.services.faq.retriever import faq_retriever

        def _do_warmup():
            logger.info("开始后台预热知识库连接与模型...")
            faq_retriever.retrieve("预热", top_k=1)
            doc_chunk_retriever.retrieve("预热", top_k=1)
            logger.info("后台预热完毕，首次检索请求不再有冷启动延迟。")

        await asyncio.to_thread(_do_warmup)
    except Exception as exc:
        logger.warning("后台预热失败（可忽略，不影响主服务）: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    apply_langsmith_env(get_settings())
    init_db()
    # 启动后台预热任务，避免阻塞 FastAPI 启动
    asyncio.create_task(warmup_indexes())
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
