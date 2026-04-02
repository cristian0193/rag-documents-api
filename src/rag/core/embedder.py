"""Sentence-Transformers embedding generation."""
import asyncio

from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper around sentence-transformers for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embedding."""
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Async embedding using thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed([text])
        return results[0]
