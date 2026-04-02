"""Text chunking with overlapping windows."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks (non-empty strings only)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]
