# CLAUDE.md — RAG Simple Project

## Project Overview

Portfolio project demonstrating a production-grade RAG (Retrieval-Augmented Generation) pipeline. Ingests PDF/TXT documents, generates embeddings, stores them in a vector database, and answers natural language questions with citations.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (async) |
| LLM | Ollama — llama3.2:3b (local, open source) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dims) |
| Vector DB | ChromaDB (embedded, cosine similarity) |
| Relational DB | PostgreSQL 16 + SQLAlchemy 2.0 async |
| Migrations | Alembic (async) |
| PDF Processing | pypdf |
| Office Processing | python-docx, openpyxl, python-pptx |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Config | pydantic-settings |
| Logging | structlog (JSON in prod, console in DEBUG) |
| Testing | pytest + pytest-asyncio |
| Infra | Docker + Docker Compose |

## Project Structure

```
src/rag/
├── main.py              # FastAPI app entry — lifespan, /health, router registration
├── config.py            # pydantic-settings — all env vars
├── schemas.py           # Pydantic request/response models
├── api/
│   ├── deps.py          # Dependency injection via app.state
│   └── routes/
│       ├── documents.py # Upload, list, get, delete endpoints
│       └── query.py     # POST /api/v1/query
├── core/
│   ├── extractor.py     # PDF/TXT/DOCX/XLSX/PPTX → raw text (registry pattern)
│   ├── chunker.py       # Text → overlapping chunks
│   ├── embedder.py      # Async sentence-transformers wrapper
│   └── llm.py           # Ollama HTTP client + RAG prompt builder
├── db/
│   ├── postgres.py      # Async engine + session factory
│   ├── models.py        # Document + Chunk ORM models
│   ├── repositories.py  # DB queries (DocumentRepository, ChunkRepository)
│   └── chroma.py        # ChromaDB async wrapper
└── services/
    ├── ingestion.py     # Orchestrates full ingestion pipeline
    └── retrieval.py     # Orchestrates query pipeline
```

## Running the Project

### With Docker (recommended)

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec app alembic upgrade head
```

Services:
- App: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Configure .env (copy from .env.example)
pytest tests/
```

## Key Architectural Decisions

- **ChromaDB embedded** — runs inside the app process (no extra service). Async safety via `asyncio.to_thread()`.
- **Services as singletons** — `Embedder`, `ChromaClient`, `OllamaClient` initialized once in lifespan, stored on `app.state`.
- **Postgres-first delete** — ON DELETE CASCADE handles chunks; Chroma cleanup is best-effort.
- **SQLAlchemy `Uuid`** (not `postgresql.UUID`) — enables SQLite in tests without driver conflicts.
- **ChromaDB IDs** use `{document_id}_{chunk_index}` format for deterministic upserts.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://rag:changeme@localhost:5432/rag_db` | PostgreSQL connection |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model name |
| `CHROMA_PATH` | `./chroma_data` | ChromaDB persistence directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | `8` | Default similar chunks to retrieve |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG = console renderer) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum file upload size |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check postgres + ollama + chroma |
| POST | `/api/v1/documents/upload` | Upload PDF or TXT |
| GET | `/api/v1/documents/` | List all documents |
| GET | `/api/v1/documents/{id}` | Document detail + chunks |
| DELETE | `/api/v1/documents/{id}` | Delete document + vectors |
| POST | `/api/v1/query` | Ask a question |

## Testing

Tests use SQLite in-memory (no Docker required). All external services are mocked.

```bash
pytest tests/                    # all tests
pytest tests/unit/               # unit only
pytest tests/integration/        # integration only
pytest -v                        # verbose output
```

## Common Commands

```bash
# View logs
docker compose logs -f app

# Re-run migrations
docker compose exec app alembic upgrade head

# Stop all services
docker compose down

# Stop + remove volumes
docker compose down -v
```
