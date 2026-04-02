from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from rag.api.routes import documents, query
from rag.config import settings


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
async def health() -> dict:
    # TODO: Phase 5 - check postgres, ollama, chroma health
    return {"status": "ok"}
