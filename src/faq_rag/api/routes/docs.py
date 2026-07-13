"""文档管理接口：入库 / 列表 / 重建索引。"""

from fastapi import APIRouter, Query

from faq_rag.exceptions import AppError
from faq_rag.models.document import Document, IngestRequest, IngestResult, ReindexResult
from faq_rag.services.doc import doc_chunk_retriever, document_store, ingest_directory

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResult)
def ingest_docs(payload: IngestRequest | None = None) -> IngestResult:
    """扫描本地 Markdown 目录，写入 MySQL 并同步文档向量索引。"""
    body = payload or IngestRequest()
    try:
        return ingest_directory(
            body.directory,
            category=body.category,
            rebuild_index=body.rebuild_index,
        )
    except FileNotFoundError as exc:
        raise AppError(str(exc), code="doc_dir_not_found") from exc


@router.get("", response_model=list[Document])
def list_docs(
    category: str | None = Query(default=None, description="按类目精确过滤"),
    enabled: bool | None = Query(default=None, description="按是否生效过滤"),
) -> list[Document]:
    return document_store.list(category=category, enabled=enabled)


@router.post("/reindex", response_model=ReindexResult)
def reindex_docs() -> ReindexResult:
    """从 MySQL 已启用文档切块全量重建 Milvus 文档索引（不重新读文件）。"""
    count = doc_chunk_retriever.sync_index()
    return ReindexResult(
        chunks_indexed=count,
        message=f"已重建文档索引，共 {count} 个切块",
    )
