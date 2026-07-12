"""In-memory FAQ store for flow validation (replace later with DB / index)."""

from datetime import datetime, timezone

from faq_rag.models.faq import FAQ, FAQCreate, FAQUpdate


class FAQStore:
    """Simple process-local store. Not for production persistence."""

    def __init__(self) -> None:
        self._items: dict[str, FAQ] = {}

    def create(self, payload: FAQCreate) -> FAQ:
        faq = FAQ.model_validate(payload.model_dump())
        self._items[faq.id] = faq
        return faq

    def list(
        self,
        *,
        q: str | None = None,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> list[FAQ]:
        items = list(self._items.values())
        if q:
            needle = q.casefold()
            items = [
                item
                for item in items
                if needle in item.question.casefold()
                or needle in item.answer.casefold()
                or any(needle in s.casefold() for s in item.similar_questions)
            ]
        if category is not None:
            items = [item for item in items if item.category == category]
        if enabled is not None:
            items = [item for item in items if item.enabled is enabled]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def get(self, faq_id: str) -> FAQ | None:
        return self._items.get(faq_id)

    def update(self, faq_id: str, payload: FAQUpdate) -> FAQ | None:
        existing = self._items.get(faq_id)
        if existing is None:
            return None
        data = existing.model_dump()
        patch = payload.model_dump(exclude_unset=True)
        data.update(patch)
        data["updated_at"] = datetime.now(timezone.utc)
        updated = FAQ.model_validate(data)
        self._items[faq_id] = updated
        return updated

    def delete(self, faq_id: str) -> bool:
        return self._items.pop(faq_id, None) is not None


faq_store = FAQStore()
