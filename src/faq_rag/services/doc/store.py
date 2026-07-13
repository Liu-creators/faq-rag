"""文档与切块持久化存储（MySQL）。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from faq_rag.db.models import DocChunkRow, DocumentRow
from faq_rag.db.session import get_session
from faq_rag.exceptions import DocumentNotFoundError
from faq_rag.models.document import DocChunk, Document


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _to_document(row: DocumentRow, *, chunk_count: int | None = None) -> Document:
    count = chunk_count
    if count is None:
        count = len(row.chunks) if row.chunks is not None else 0
    return Document(
        id=row.id,
        title=row.title,
        source_path=row.source_path,
        source_url=row.source_url,
        category=row.category,
        enabled=bool(row.enabled),
        chunk_count=count,
        created_at=_ensure_aware(row.created_at),
        updated_at=_ensure_aware(row.updated_at),
    )


def _to_chunk(row: DocChunkRow) -> DocChunk:
    return DocChunk(
        id=row.id,
        document_id=row.document_id,
        chunk_index=row.chunk_index,
        heading=row.heading,
        content=row.content,
        token_estimate=row.token_estimate,
        created_at=_ensure_aware(row.created_at),
        updated_at=_ensure_aware(row.updated_at),
    )


class DocumentStore:
    """文档元数据 + 切块 CRUD。"""

    def list(
        self,
        *,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> list[Document]:
        stmt = select(DocumentRow).options(selectinload(DocumentRow.chunks))
        if category is not None:
            stmt = stmt.where(DocumentRow.category == category)
        if enabled is not None:
            stmt = stmt.where(DocumentRow.enabled.is_(enabled))
        stmt = stmt.order_by(DocumentRow.created_at.desc())

        with get_session() as session:
            rows = list(session.scalars(stmt).unique().all())
            return [
                _to_document(row, chunk_count=len(row.chunks)) for row in rows
            ]

    def get(self, document_id: str) -> Document:
        doc = self.find(document_id)
        if doc is None:
            raise DocumentNotFoundError(document_id)
        return doc

    def find(self, document_id: str) -> Document | None:
        with get_session() as session:
            row = session.scalars(
                select(DocumentRow)
                .where(DocumentRow.id == document_id)
                .options(selectinload(DocumentRow.chunks))
            ).first()
            if row is None:
                return None
            return _to_document(row, chunk_count=len(row.chunks))

    def find_by_source_path(self, source_path: str) -> Document | None:
        stmt = (
            select(DocumentRow)
            .where(DocumentRow.source_path == source_path)
            .options(selectinload(DocumentRow.chunks))
        )
        with get_session() as session:
            row = session.scalars(stmt).first()
            if row is None:
                return None
            return _to_document(row, chunk_count=len(row.chunks))

    def upsert_document(
        self,
        *,
        title: str,
        source_path: str | None = None,
        source_url: str | None = None,
        category: str | None = None,
        enabled: bool = True,
        document_id: str | None = None,
    ) -> Document:
        """按 id 或 source_path 更新；否则新建。"""
        now = _utc_now()
        with get_session() as session:
            row: DocumentRow | None = None
            if document_id:
                row = session.get(DocumentRow, document_id)
            if row is None and source_path:
                row = session.scalars(
                    select(DocumentRow).where(DocumentRow.source_path == source_path)
                ).first()

            if row is None:
                row = DocumentRow(
                    id=str(uuid4()),
                    title=title,
                    source_path=source_path,
                    source_url=source_url,
                    category=category,
                    enabled=enabled,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.title = title
                row.source_path = source_path
                row.source_url = source_url
                if category is not None:
                    row.category = category
                row.enabled = enabled
                row.updated_at = now

            session.flush()
            count = session.scalar(
                select(func.count())
                .select_from(DocChunkRow)
                .where(DocChunkRow.document_id == row.id)
            )
            return _to_document(row, chunk_count=int(count or 0))

    def replace_chunks(
        self,
        document_id: str,
        chunks: list[tuple[str | None, str]],
    ) -> list[DocChunk]:
        """替换某文档下全部切块。返回新切块列表。

        ``chunks`` 元素为 ``(heading, content)``。
        """
        now = _utc_now()
        with get_session() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                raise DocumentNotFoundError(document_id)

            session.execute(
                delete(DocChunkRow).where(DocChunkRow.document_id == document_id)
            )

            created: list[DocChunkRow] = []
            for index, (heading, content) in enumerate(chunks):
                text = content.strip()
                if not text:
                    continue
                chunk_row = DocChunkRow(
                    id=str(uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    heading=heading,
                    content=text,
                    token_estimate=len(text),
                    created_at=now,
                    updated_at=now,
                )
                session.add(chunk_row)
                created.append(chunk_row)

            row.updated_at = now
            session.flush()
            return [_to_chunk(item) for item in created]

    def list_chunks(
        self,
        *,
        document_id: str | None = None,
        enabled_docs_only: bool = False,
    ) -> list[DocChunk]:
        stmt = select(DocChunkRow)
        if document_id is not None:
            stmt = stmt.where(DocChunkRow.document_id == document_id)
        if enabled_docs_only:
            stmt = stmt.join(DocumentRow).where(DocumentRow.enabled.is_(True))
        stmt = stmt.order_by(DocChunkRow.document_id, DocChunkRow.chunk_index)

        with get_session() as session:
            rows = list(session.scalars(stmt).all())
            return [_to_chunk(row) for row in rows]

    def find_chunk(self, chunk_id: str) -> DocChunk | None:
        with get_session() as session:
            row = session.get(DocChunkRow, chunk_id)
            if row is None:
                return None
            return _to_chunk(row)

    def list_chunks_with_documents(
        self,
        *,
        enabled_docs_only: bool = True,
    ) -> list[tuple[DocChunk, Document]]:
        """返回 (chunk, document) 对，供检索水合。"""
        stmt = select(DocChunkRow, DocumentRow).join(
            DocumentRow, DocChunkRow.document_id == DocumentRow.id
        )
        if enabled_docs_only:
            stmt = stmt.where(DocumentRow.enabled.is_(True))
        stmt = stmt.order_by(DocChunkRow.document_id, DocChunkRow.chunk_index)

        with get_session() as session:
            pairs = list(session.execute(stmt).all())
            return [
                (_to_chunk(chunk_row), _to_document(doc_row, chunk_count=0))
                for chunk_row, doc_row in pairs
            ]

    def delete(self, document_id: str) -> list[str]:
        """删除文档及其切块；返回被删 chunk id 列表（便于清索引）。"""
        with get_session() as session:
            row = session.scalars(
                select(DocumentRow)
                .where(DocumentRow.id == document_id)
                .options(selectinload(DocumentRow.chunks))
            ).first()
            if row is None:
                raise DocumentNotFoundError(document_id)
            chunk_ids = [chunk.id for chunk in row.chunks]
            session.delete(row)
            return chunk_ids


document_store = DocumentStore()
