"""FAQ 持久化存储（MySQL）。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select

from faq_rag.db.models import FAQRow
from faq_rag.db.session import get_session
from faq_rag.exceptions import FAQNotFoundError
from faq_rag.models.faq import FAQ, FAQCreate, FAQUpdate


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _to_schema(row: FAQRow) -> FAQ:
    similar = row.similar_questions if isinstance(row.similar_questions, list) else []
    return FAQ(
        id=row.id,
        question=row.question,
        answer=row.answer,
        similar_questions=[str(item) for item in similar],
        category=row.category,
        enabled=bool(row.enabled),
        created_at=_ensure_aware(row.created_at),
        updated_at=_ensure_aware(row.updated_at),
    )


class FAQStore:
    """基于 MySQL 的 FAQ CRUD。对外 API 与原先内存版保持一致。"""

    def create(self, payload: FAQCreate) -> FAQ:
        now = _utc_now()
        row = FAQRow(
            id=str(uuid4()),
            question=payload.question,
            answer=payload.answer,
            similar_questions=list(payload.similar_questions),
            category=payload.category,
            enabled=payload.enabled,
            created_at=now,
            updated_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
            return _to_schema(row)

    def list(
        self,
        *,
        q: str | None = None,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> list[FAQ]:
        stmt = select(FAQRow)
        if category is not None:
            stmt = stmt.where(FAQRow.category == category)
        if enabled is not None:
            stmt = stmt.where(FAQRow.enabled.is_(enabled))
        stmt = stmt.order_by(FAQRow.created_at.desc())

        with get_session() as session:
            rows = list(session.scalars(stmt).all())

        items = [_to_schema(row) for row in rows]
        if q:
            needle = q.casefold()
            items = [
                item
                for item in items
                if needle in item.question.casefold()
                or needle in item.answer.casefold()
                or any(needle in s.casefold() for s in item.similar_questions)
            ]
        return items

    def get(self, faq_id: str) -> FAQ:
        faq = self.find(faq_id)
        if faq is None:
            raise FAQNotFoundError(faq_id)
        return faq

    def find(self, faq_id: str) -> FAQ | None:
        with get_session() as session:
            row = session.get(FAQRow, faq_id)
            if row is None:
                return None
            return _to_schema(row)

    def update(self, faq_id: str, payload: FAQUpdate) -> FAQ:
        with get_session() as session:
            row = session.get(FAQRow, faq_id)
            if row is None:
                raise FAQNotFoundError(faq_id)

            patch = payload.model_dump(exclude_unset=True)
            if "question" in patch:
                row.question = patch["question"]
            if "answer" in patch:
                row.answer = patch["answer"]
            if "similar_questions" in patch:
                row.similar_questions = list(patch["similar_questions"] or [])
            if "category" in patch:
                row.category = patch["category"]
            if "enabled" in patch:
                row.enabled = bool(patch["enabled"])
            row.updated_at = _utc_now()
            session.flush()
            return _to_schema(row)

    def delete(self, faq_id: str) -> None:
        with get_session() as session:
            row = session.get(FAQRow, faq_id)
            if row is None:
                raise FAQNotFoundError(faq_id)
            session.delete(row)


faq_store = FAQStore()
