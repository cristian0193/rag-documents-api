"""Document CRUD endpoints."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.deps import get_db, get_chroma, get_ingestion_service
from rag.config import settings
from rag.db.chroma import ChromaClient
from rag.db.repositories import DocumentRepository, ChunkRepository
from rag.schemas import DocumentResponse, DocumentDetailResponse, ChunkResponse
from rag.services.ingestion import IngestionService
from rag.core.extractor import UnsupportedFileTypeError

router = APIRouter()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentResponse:
    """Upload a PDF or TXT document for ingestion."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must have a filename.")
    try:
        MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024
        file_bytes = await file.read(MAX_BYTES + 1)
        if len(file_bytes) > MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
            )
        document = await service.ingest(file_bytes, file.filename, db)
        return DocumentResponse.model_validate(document)
    except HTTPException:
        raise
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document ingestion.",
        )


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentResponse]:
    """List all ingested documents."""
    repo = DocumentRepository(db)
    documents = await repo.list_all()
    return [DocumentResponse.model_validate(doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Get document details with chunks."""
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    chunks = await chunk_repo.get_by_document_id(document_id)
    chunk_responses = [ChunkResponse.model_validate(c) for c in chunks]

    return DocumentDetailResponse(
        **DocumentResponse.model_validate(document).model_dump(),
        chunks=chunk_responses,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    chroma: ChromaClient = Depends(get_chroma),
) -> None:
    """Delete a document and all its chunks from both DBs."""
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    # Fetch chroma IDs before deleting from postgres
    chroma_ids = await chunk_repo.get_chroma_ids_by_document(document_id)

    # Delete from postgres first (cascades chunks via FK)
    deleted = await doc_repo.delete(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Best-effort cleanup of chroma vectors
    if chroma_ids:
        try:
            await chroma.delete_by_ids(chroma_ids)
        except Exception:
            pass  # Orphan vectors are preferable to a failed delete response
