from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql+asyncpg://rag:changeme@localhost:5432/rag_db"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ChromaDB
    chroma_path: str = "./chroma_data"
    chroma_collection: str = "rag_documents"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    default_top_k: int = 5

    # App
    log_level: str = "INFO"
    max_upload_size_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias — uses the cached instance
settings = get_settings()
