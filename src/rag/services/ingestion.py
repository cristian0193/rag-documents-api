"""Document ingestion orchestration service."""
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from rag.config import settings
from rag.core.chunker import chunk_text
from rag.core.embedder import Embedder
from rag.core.extractor import extract_text, validate_file_type
from rag.db.chroma import ChromaClient
from rag.db.models import Document
from rag.db.repositories import ChunkRepository, DocumentRepository

logger = structlog.get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        chroma: ChromaClient,
    ):
        self.embedder = embedder
        self.chroma = chroma

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        db: AsyncSession,
    ) -> Document:
        """
        Full ingestion pipeline:
        1. Create document record in PostgreSQL (status='processing')
        2. Extract text from file
        3. Chunk the text
        4. Generate embeddings for all chunks
        5. Store embeddings + documents in ChromaDB
        6. Store chunk metadata in PostgreSQL
        7. Update document status to 'ready' with total_chunks count
        8. Return the updated Document

        On any error: update document status to 'error' with error message, then re-raise
        """
        validate_file_type(filename)  # raises UnsupportedFileTypeError before any DB writes

        file_type = Path(filename).suffix.lstrip(".").lower()
        file_size = len(file_bytes)

        logger.info("ingestion.started", filename=filename, file_size=len(file_bytes))

        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)

        # Step 1: Create document record with status='processing'
        document = await doc_repo.create(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
        )
        document_id = document.id

        chroma_ids_written: list[str] = []

        try:
            # Step 2: Extract text from file
            text = extract_text(file_bytes, filename)
            logger.info("ingestion.extracted", filename=filename, text_length=len(text))

            # Step 3: Chunk the text
            chunks = chunk_text(
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            logger.info("ingestion.chunked", filename=filename, chunk_count=len(chunks))

            # Step 4: Generate embeddings for all chunks
            embeddings = await self.embedder.embed(chunks)
            logger.info("ingestion.embedded", filename=filename, vector_count=len(embeddings))

            # Step 5: Store embeddings + documents in ChromaDB
            chroma_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            await self.chroma.upsert(
                ids=chroma_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            chroma_ids_written = chroma_ids
            logger.info("ingestion.stored_chroma", document_id=str(document_id), chunk_count=len(chunks))

            # Step 6: Store chunk metadata in PostgreSQL
            chunks_data = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk,
                    "token_count": len(chunk.split()),  # approximation: word count
                    "chroma_id": chroma_ids[i],
                }
                for i, chunk in enumerate(chunks)
            ]
            await chunk_repo.create_batch(chunks_data)

            # Step 7: Update document status to 'ready' with total_chunks count
            await doc_repo.update_ready(document_id, total_chunks=len(chunks))
            await db.commit()

            # Step 8: Return the updated Document
            document = await doc_repo.get_by_id(document_id)
            logger.info("ingestion.complete", document_id=str(document_id), filename=filename, total_chunks=len(chunks))
            return document

        except Exception as e:
            logger.error("ingestion.failed", filename=filename, error=str(e))
            if chroma_ids_written:
                try:
                    await self.chroma.delete_by_ids(chroma_ids_written)
                except Exception:
                    pass  # best-effort cleanup
            await doc_repo.update_status(document_id, status="error", error_msg=str(e))
            await db.commit()
            raise
