"""RAG query endpoint."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag.api.deps import get_db, get_retrieval_service
from rag.core.llm import OllamaError
from rag.schemas import QueryRequest, QueryResponse
from rag.services.retrieval import RetrievalService

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """Query documents with a natural language question."""
    try:
        return await service.query(request.question, request.top_k, db)
    except OllamaError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {e}",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing.",
        )
