"""FAQ CRUD 接口（MySQL 持久化 + 相似度索引增量同步）。"""

from fastapi import APIRouter, Query, status

from faq_rag.models.faq import FAQ, FAQCreate, FAQUpdate
from faq_rag.services.faq import faq_retriever, faq_store

router = APIRouter(prefix="/faqs", tags=["faqs"])


@router.post("", response_model=FAQ, status_code=status.HTTP_201_CREATED)
def create_faq(payload: FAQCreate) -> FAQ:
    faq = faq_store.create(payload)
    faq_retriever.upsert_faq(faq)
    return faq


@router.get("", response_model=list[FAQ])
def list_faqs(
    q: str | None = Query(default=None, description="按标准问/答案/相似问关键词过滤"),
    category: str | None = Query(default=None, description="按类目精确过滤"),
    enabled: bool | None = Query(default=None, description="按是否生效过滤"),
) -> list[FAQ]:
    return faq_store.list(q=q, category=category, enabled=enabled)


@router.get("/{faq_id}", response_model=FAQ)
def get_faq(faq_id: str) -> FAQ:
    return faq_store.get(faq_id)


@router.patch("/{faq_id}", response_model=FAQ)
def update_faq(faq_id: str, payload: FAQUpdate) -> FAQ:
    patch = payload.model_dump(exclude_unset=True)
    faq = faq_store.update(faq_id, payload)
    if faq_retriever.index_fields_changed(patch):
        faq_retriever.upsert_faq(faq)
    return faq


@router.delete("/{faq_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_faq(faq_id: str) -> None:
    faq_store.delete(faq_id)
    faq_retriever.remove_faq(faq_id)
