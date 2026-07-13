"""提问流水线：FAQ 闸门、改写与文档 RAG 兜底。"""

from faq_rag.services.ask.pipeline import AskPipeline, ask_pipeline

__all__ = [
    "AskPipeline",
    "ask_pipeline",
]
