"""文档 / 切块 CRUD 与检索用 schema。"""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Document(BaseModel):
    """已存储的文档元数据。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_path: str | None = None
    source_url: str | None = None
    category: str | None = None
    enabled: bool = True
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class DocChunk(BaseModel):
    """文档切块。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    heading: str | None = None
    content: str
    token_estimate: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class IngestRequest(BaseModel):
    """从本地目录入库文档。"""

    directory: str | None = Field(
        default=None,
        description="Markdown 根目录；默认使用 DOC_RAW_DIR",
    )
    category: str | None = Field(default=None, description="可选类目，写入全部新文档")
    rebuild_index: bool = Field(
        default=True,
        description="入库后是否全量重建文档向量索引",
    )


class IngestResult(BaseModel):
    documents_upserted: int
    chunks_written: int
    directory: str
    index_rebuilt: bool


class ReindexResult(BaseModel):
    chunks_indexed: int
    message: str
