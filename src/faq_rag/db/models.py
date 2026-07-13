"""ORM 表定义。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

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
