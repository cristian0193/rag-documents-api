"""Unit tests for rag.core.embedder.Embedder (SentenceTransformer mocked)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model that returns fixed-shape arrays."""
    model = MagicMock()
    # Default: return two 384-dim vectors
    model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
    return model


@pytest.fixture
def embedder(mock_model):
    """Embedder instance with SentenceTransformer replaced by mock_model."""
    with patch("rag.core.embedder.SentenceTransformer", return_value=mock_model):
        from rag.core.embedder import Embedder
        return Embedder("test-model")


class TestEmbedder:
    async def test_embed_returns_list_of_lists(self, embedder, mock_model):
        """embed() returns list[list[float]] with shape (n_texts, dim)."""
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        result = await embedder.embed(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    async def test_embed_query_returns_single_vector(self, embedder, mock_model):
        """embed_query() returns a single list of floats with the expected dimension."""
        mock_model.encode.return_value = np.array([[0.5] * 384])

        result = await embedder.embed_query("What is RAG?")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    def test_embed_sync_returns_correct_shape(self, embedder, mock_model):
        """embed_sync() returns list[list[float]] without blocking the event loop."""
        mock_model.encode.return_value = np.array([[0.3] * 384])

        result = embedder.embed_sync(["single text"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_embed_sync_calls_model_encode(self, embedder, mock_model):
        """embed_sync() delegates to model.encode with convert_to_numpy=True."""
        mock_model.encode.return_value = np.array([[0.0] * 384])

        embedder.embed_sync(["hello"])

        mock_model.encode.assert_called_once_with(["hello"], convert_to_numpy=True)

    async def test_embed_single_text(self, embedder, mock_model):
        """embed() with a single-element list returns a list with one vector."""
        mock_model.encode.return_value = np.array([[0.7] * 384])

        result = await embedder.embed(["only one"])

        assert len(result) == 1
        assert len(result[0]) == 384
