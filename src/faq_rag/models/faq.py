"""FAQ schemas for CRUD and later retrieval."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FAQCreate(BaseModel):
    """Payload for creating a FAQ entry."""

    question: str = Field(..., min_length=1, description="标准问法")
    answer: str = Field(..., min_length=1, description="标准答案")
    similar_questions: list[str] = Field(
        default_factory=list,
        description="相似问法，后续用于检索扩展",
    )
    category: str | None = Field(default=None, description="可选类目")
    enabled: bool = Field(default=True, description="是否生效")


class FAQUpdate(BaseModel):
    """Partial update payload; omitted fields are left unchanged."""

    question: str | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    similar_questions: list[str] | None = None
    category: str | None = None
    enabled: bool | None = None


class FAQ(FAQCreate):
    """Stored FAQ record."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
