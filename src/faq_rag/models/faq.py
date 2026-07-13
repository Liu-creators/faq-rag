"""FAQ 的 CRUD 与后续检索用 schema。"""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FAQCreate(BaseModel):
    """创建 FAQ 条目的请求体。"""

    question: str = Field(..., min_length=1, description="标准问法")
    answer: str = Field(..., min_length=1, description="标准答案")
    similar_questions: list[str] = Field(
        default_factory=list,
        description="相似问法，后续用于检索扩展",
    )
    category: str | None = Field(default=None, description="可选类目")
    enabled: bool = Field(default=True, description="是否生效")


class FAQUpdate(BaseModel):
    """部分更新请求体；未传字段保持不变。"""

    question: str | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    similar_questions: list[str] | None = None
    category: str | None = None
    enabled: bool | None = None


class FAQ(FAQCreate):
    """已存储的 FAQ 记录。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
