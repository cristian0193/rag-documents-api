import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    file_size: int | None
    total_chunks: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    chunk_index: int
    text: str
    token_count: int | None
    chroma_id: str


class DocumentDetailResponse(DocumentResponse):
    chunks: list[ChunkResponse] = []


class SourceChunk(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
