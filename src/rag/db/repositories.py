import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.models import Chunk, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, filename: str, file_type: str, file_size: int | None) -> Document:
        doc = Document(filename=filename, file_type=file_type, file_size=file_size)
        self.session.add(doc)
        await self.session.flush()
        await self.session.refresh(doc)
        return doc

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self, document_id: uuid.UUID, status: str, error_msg: str | None = None
    ) -> None:
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(status=status, error_msg=error_msg)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)

    async def update_chunk_count(self, document_id: uuid.UUID, total_chunks: int) -> None:
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(total_chunks=total_chunks)
            .execution_options(synchronize_session="fetch")
        )
        await self.session.execute(stmt)

    async def delete(self, document_id: uuid.UUID) -> bool:
        stmt = delete(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_batch(self, chunks_data: list[dict]) -> list[Chunk]:
        chunks = [Chunk(**data) for data in chunks_data]
        self.session.add_all(chunks)
        await self.session.flush()  # assigns DB-generated values
        return chunks

    async def get_by_document_id(self, document_id: uuid.UUID) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_chroma_ids_by_document(self, document_id: uuid.UUID) -> list[str]:
        stmt = select(Chunk.chroma_id).where(Chunk.document_id == document_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
