import asyncio

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.config import settings


class ChromaClient:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        await asyncio.to_thread(
            self._collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict:
        return await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    async def delete_by_ids(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._collection.delete, ids=ids)

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)
