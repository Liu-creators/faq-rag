"""ORM 表定义。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from faq_rag.db.base import Base


class FAQRow(Base):
    """FAQ 业务表：标准问 / 答案 / 相似问法。

    向量库（Milvus）只索引问法文本；标准答与元数据以本表为准。
    """

    __tablename__ = "faqs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    similar_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )


class DocumentRow(Base):
    """文档元数据表；正文以切块形式存在 ``doc_chunks``。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )

    chunks: Mapped[list[DocChunkRow]] = relationship(
        "DocChunkRow",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocChunkRow.chunk_index",
    )


class DocChunkRow(Base):
    """文档切块：Milvus ``doc_chunks`` 集合的 ``doc_id`` 对本表 ``id``。"""

    __tablename__ = "doc_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )

    document: Mapped[DocumentRow] = relationship("DocumentRow", back_populates="chunks")
