"""Unit tests for rag.core.extractor.extract_text."""
import pytest

pytestmark = pytest.mark.asyncio

from rag.core.extractor import extract_text, UnsupportedFileTypeError


class TestExtractTxt:
    def test_extract_txt_utf8(self):
        """UTF-8 encoded TXT bytes are decoded and returned as-is."""
        content = "Hello world"
        result = extract_text(content.encode("utf-8"), "doc.txt")
        assert "Hello world" in result

    def test_extract_txt_latin1(self):
        """Latin-1 encoded bytes (with accented chars) don't crash and return a string."""
        # 'café' encoded in latin-1 — not valid UTF-8
        content = "café résumé".encode("latin-1")
        result = extract_text(content, "accented.txt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_txt_strips_whitespace(self):
        """Runs of 3+ consecutive newlines are collapsed to at most 2."""
        content = "Line one\n\n\n\n\nLine two\n\n\n\nLine three"
        result = extract_text(content.encode("utf-8"), "spaced.txt")
        # There must be no run of 3+ newlines in the result
        assert "\n\n\n" not in result

    def test_extract_txt_leading_trailing_stripped(self):
        """Leading and trailing whitespace is stripped from the result."""
        content = "   \n\nHello\n\n   "
        result = extract_text(content.encode("utf-8"), "padded.txt")
        assert result == result.strip()


class TestExtractPdf:
    def test_extract_pdf(self, sample_pdf_bytes):
        """A minimal valid PDF returns a non-empty string."""
        result = extract_text(sample_pdf_bytes, "sample.pdf")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_extract_pdf_contains_text(self, sample_pdf_bytes):
        """The minimal PDF fixture contains 'Hello from PDF'."""
        result = extract_text(sample_pdf_bytes, "sample.pdf")
        assert "Hello" in result


class TestUnsupportedFileType:
    def test_unsupported_file_type_exe(self):
        """A .exe extension raises UnsupportedFileTypeError."""
        with pytest.raises(UnsupportedFileTypeError):
            extract_text(b"\x00\x01\x02", "binary.exe")

    def test_unsupported_file_type_csv(self):
        """A .csv extension raises UnsupportedFileTypeError."""
        with pytest.raises(UnsupportedFileTypeError):
            extract_text(b"a,b,c\n1,2,3", "data.csv")

    def test_unsupported_file_type_no_extension(self):
        """A file with no recognised extension raises UnsupportedFileTypeError."""
        with pytest.raises(UnsupportedFileTypeError):
            extract_text(b"some data", "noextension")
