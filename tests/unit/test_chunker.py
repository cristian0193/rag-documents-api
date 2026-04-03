"""Unit tests for rag.core.chunker.chunk_text."""
import pytest

pytestmark = pytest.mark.asyncio

from rag.core.chunker import chunk_text


class TestChunkText:
    def test_chunk_basic(self):
        """A 1000-character text returns a list of non-empty strings."""
        text = "A" * 1000
        result = chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)
        assert all(len(chunk) > 0 for chunk in result)

    def test_chunk_overlap_produces_multiple_chunks(self):
        """A text long enough for multiple chunks actually yields more than one."""
        # ~1350 chars with default chunk_size=512 and overlap=50 → at least 2 chunks
        text = "The quick brown fox jumps over the lazy dog. " * 30
        result = chunk_text(text, chunk_size=512, chunk_overlap=50)
        assert len(result) > 1

    def test_chunk_empty_input(self):
        """An empty string returns an empty list."""
        result = chunk_text("")
        assert result == []

    def test_chunk_short_text(self):
        """Text shorter than chunk_size returns a single chunk."""
        text = "Short text."
        result = chunk_text(text, chunk_size=512)
        assert len(result) == 1
        assert result[0].strip() == text.strip()

    def test_chunk_no_empty_strings(self):
        """No chunk in the result is empty or whitespace-only."""
        text = "Word " * 500  # ~2500 chars
        result = chunk_text(text, chunk_size=512, chunk_overlap=50)
        for chunk in result:
            assert chunk.strip() != "", f"Found empty chunk: {repr(chunk)}"

    def test_chunk_custom_size(self):
        """Custom chunk_size is respected (no chunk exceeds it by large margin)."""
        text = "Hello world! " * 100  # ~1300 chars
        chunk_size = 100
        result = chunk_text(text, chunk_size=chunk_size, chunk_overlap=10)
        # RecursiveCharacterTextSplitter may slightly exceed chunk_size at word
        # boundaries, but no chunk should be wildly larger than requested
        for chunk in result:
            assert len(chunk) <= chunk_size * 2, (
                f"Chunk too long ({len(chunk)} chars): {chunk[:50]!r}"
            )

    def test_chunk_whitespace_only_input(self):
        """Whitespace-only input returns an empty list (all chunks filtered out)."""
        result = chunk_text("   \n\n   \t   ")
        assert result == []
