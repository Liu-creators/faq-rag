"""FastAPI application entrypoint."""

from fastapi import FastAPI

from faq_rag import __version__
from faq_rag.api.routes import api_router

app = FastAPI(
    title="FAQ RAG",
    description="FAQ retrieval-augmented generation API",
    version=__version__,
)

app.include_router(api_router)


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    return {"message": "FAQ RAG API is running", "docs": "/docs"}


def run() -> None:
    """Start the development server."""
    import uvicorn

    uvicorn.run(
        "faq_rag.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
