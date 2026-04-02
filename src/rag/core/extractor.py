"""Text extraction from PDF and TXT files."""
import io
import re
from pathlib import Path

import pypdf


class UnsupportedFileTypeError(ValueError):
    pass


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract raw text from file bytes.

    Args:
        file_bytes: Raw file content
        filename: Original filename (used to determine file type)

    Returns:
        Extracted text as a string

    Raises:
        UnsupportedFileTypeError: If file type is not PDF or TXT
        ValueError: If PDF is empty or unreadable
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

    elif suffix == ".pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        except pypdf.errors.PdfReadError as e:
            raise ValueError(f"Failed to read PDF '{filename}': {e}") from e
        if len(reader.pages) == 0:
            raise ValueError(f"PDF '{filename}' is empty or unreadable.")
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        text = "\n".join(pages_text)
        if not text.strip():
            raise ValueError(f"Could not extract text from PDF '{filename}'. The file may be image-only or encrypted.")

    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Only .pdf and .txt are supported."
        )

    # Collapse runs of 3+ newlines down to 2, and strip leading/trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
