"""Shared FastAPI dependency injection — DB sessions and service singletons."""
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.postgres import get_db
from rag.db.chroma import ChromaClient
from rag.core.embedder import Embedder
from rag.core.llm import OllamaClient
from rag.services.ingestion import IngestionService
from rag.services.retrieval import RetrievalService


def get_chroma(request: Request) -> ChromaClient:
    return request.app.state.chroma


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_llm(request: Request) -> OllamaClient:
    return request.app.state.llm


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service
