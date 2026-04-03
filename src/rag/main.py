from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from rag.api.routes import documents, query
from rag.config import settings
from rag.core.embedder import Embedder
from rag.core.llm import OllamaClient
from rag.db.chroma import ChromaClient
from rag.db.postgres import AsyncSessionLocal
from rag.services.ingestion import IngestionService
from rag.services.retrieval import RetrievalService


def configure_structlog(log_level: str = "INFO") -> None:
    renderer = (
        structlog.dev.ConsoleRenderer()
        if log_level.upper() == "DEBUG"
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level.upper(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


configure_structlog(settings.log_level)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize singleton services
    chroma = ChromaClient()
    embedder = Embedder(model_name=settings.embedding_model)
    llm = OllamaClient(base_url=settings.ollama_url, model=settings.ollama_model)

    app.state.chroma = chroma
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.ingestion_service = IngestionService(embedder=embedder, chroma=chroma)
    app.state.retrieval_service = RetrievalService(embedder=embedder, chroma=chroma, llm=llm)

    logger.info("RAG Simple API starting up", version="0.1.0")
    yield
    logger.info("RAG Simple API shutting down")


app = FastAPI(
    title="RAG Simple API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Check health of all services."""
    llm: OllamaClient = request.app.state.llm
    chroma: ChromaClient = request.app.state.chroma

    # Check postgres
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        postgres_ok = False

    # Check ollama
    ollama_ok = await llm.health_check()

    # Check chroma
    try:
        await chroma.count()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    all_ok = postgres_ok and ollama_ok and chroma_ok
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={
            "postgres": "ok" if postgres_ok else "error",
            "ollama": "ok" if ollama_ok else "error",
            "chroma": "ok" if chroma_ok else "error",
        },
        status_code=status_code,
    )
