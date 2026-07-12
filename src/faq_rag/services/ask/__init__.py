"""Ask pipeline: FAQ gate, rewrite, and document RAG fallback."""

from faq_rag.services.ask.pipeline import AskPipeline, ask_pipeline

__all__ = [
    "AskPipeline",
    "ask_pipeline",
]
