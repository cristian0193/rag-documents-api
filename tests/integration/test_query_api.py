"""Integration tests for the RAG query API (/api/v1/query/)."""
import uuid
import pytest
from unittest.mock import AsyncMock

pytestmark = pytest.mark.asyncio

from rag.core.llm import OllamaError


class TestQueryEndpoint:
    async def test_query_returns_answer(self, client):
        """POST /api/v1/query/ with a valid question returns 200, answer, and sources."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "What is this about?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    async def test_query_answer_is_string(self, client):
        """The answer field in the response is a non-empty string."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "Explain the content.", "top_k": 3},
        )
        assert response.status_code == 200
        assert isinstance(response.json()["answer"], str)
        assert len(response.json()["answer"]) > 0

    async def test_query_sources_structure(self, client):
        """Each source in the response has the expected fields."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "Tell me more.", "top_k": 5},
        )
        assert response.status_code == 200
        sources = response.json()["sources"]
        if sources:  # may be empty if chroma returns no results
            source = sources[0]
            assert "document_id" in source
            assert "filename" in source
            assert "chunk_index" in source
            assert "text" in source
            assert "score" in source


class TestQueryValidation:
    async def test_query_validation_empty_question(self, client):
        """An empty question string is rejected with 422 (Pydantic min_length=1)."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "", "top_k": 5},
        )
        assert response.status_code == 422

    async def test_query_validation_top_k_too_large(self, client):
        """top_k > 20 violates the le=20 constraint and returns 422."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "test", "top_k": 100},
        )
        assert response.status_code == 422

    async def test_query_validation_top_k_zero(self, client):
        """top_k < 1 violates the ge=1 constraint and returns 422."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

    async def test_query_validation_missing_question(self, client):
        """Omitting the required 'question' field returns 422."""
        response = await client.post(
            "/api/v1/query/",
            json={"top_k": 5},
        )
        assert response.status_code == 422

    async def test_query_default_top_k(self, client):
        """Omitting top_k uses the default (5) and still returns 200."""
        response = await client.post(
            "/api/v1/query/",
            json={"question": "What is here?"},
        )
        assert response.status_code == 200


class TestQueryOllamaUnavailable:
    async def test_query_ollama_unavailable(self, client, mock_llm):
        """When OllamaClient.generate raises OllamaError, the endpoint returns 503."""
        mock_llm.generate = AsyncMock(side_effect=OllamaError("unavailable"))

        # Rebuild the retrieval service so it picks up the updated mock
        from rag.services.retrieval import RetrievalService
        from rag.services.ingestion import IngestionService

        client.app.state.retrieval_service = RetrievalService(
            embedder=client.app.state.embedder,
            chroma=client.app.state.chroma,
            llm=mock_llm,
        )

        response = await client.post(
            "/api/v1/query/",
            json={"question": "Will this fail?", "top_k": 3},
        )
        assert response.status_code == 503
