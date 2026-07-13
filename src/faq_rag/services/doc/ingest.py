"""从本地 Markdown 目录入库文档并同步向量索引。"""

from __future__ import annotations

import logging
from pathlib import Path

from faq_rag.config import get_settings
from faq_rag.models.document import IngestResult
from faq_rag.services.doc.chunker import chunk_markdown
from faq_rag.services.doc.store import DocumentStore, document_store

logger = logging.getLogger(__name__)


def _title_from_markdown(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or path.stem
    return path.stem


def ingest_directory(
    directory: str | Path | None = None,
    *,
    category: str | None = None,
    rebuild_index: bool = True,
    store: DocumentStore | None = None,
    retriever=None,
) -> IngestResult:
    """扫描目录下 ``*.md``，按 ``source_path`` upsert 文档并替换切块。

    ``retriever`` 为 ``DocChunkRetriever``；默认延迟导入单例，避免循环依赖。
    """
    from faq_rag.services.doc.retriever import DocChunkRetriever, doc_chunk_retriever

    settings = get_settings()
    root = Path(directory or settings.doc_raw_dir).expanduser().resolve()
    doc_store = store or document_store
    index: DocChunkRetriever = retriever or doc_chunk_retriever

    if not root.is_dir():
        raise FileNotFoundError(f"文档目录不存在：{root}")

    md_files = sorted(root.rglob("*.md"))
    documents_upserted = 0
    chunks_written = 0

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        title = _title_from_markdown(path, text)
        source_path = str(path)
        relative = path.relative_to(root).as_posix()

        # 先找旧文档，便于清理旧 chunk 索引
        existing = doc_store.find_by_source_path(source_path)
        old_chunk_ids: list[str] = []
        if existing is not None:
            old_chunk_ids = [
                c.id for c in doc_store.list_chunks(document_id=existing.id)
            ]

        doc = doc_store.upsert_document(
            title=title,
            source_path=source_path,
            source_url=None,
            category=category,
            enabled=True,
            document_id=existing.id if existing else None,
        )
        pieces = chunk_markdown(text)
        new_chunks = doc_store.replace_chunks(
            doc.id,
            [(piece.heading, piece.content) for piece in pieces],
        )
        documents_upserted += 1
        chunks_written += len(new_chunks)

        if not rebuild_index:
            for chunk_id in old_chunk_ids:
                index.remove_chunk(chunk_id)
            index.upsert_chunks(new_chunks)

        logger.info(
            "入库 %s → 文档 %s，%d 个切块",
            relative,
            doc.id,
            len(new_chunks),
        )

    index_rebuilt = False
    if rebuild_index:
        index.sync_index()
        index_rebuilt = True

    return IngestResult(
        documents_upserted=documents_upserted,
        chunks_written=chunks_written,
        directory=str(root),
        index_rebuilt=index_rebuilt,
    )
