"""Integration tests for the document ingestion API (/api/v1/documents/)."""
import io
import uuid
import pytest

pytestmark = pytest.mark.asyncio


class TestUploadDocument:
    async def test_upload_txt_document(self, client, sample_txt_bytes):
        """Uploading a valid TXT file returns 201 with document metadata."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample.txt", io.BytesIO(sample_txt_bytes), "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["filename"] == "sample.txt"
        assert "status" in data
        # UUID must be parseable
        uuid.UUID(data["id"])

    async def test_upload_invalid_file_type(self, client):
        """Uploading an unsupported file type returns 400."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("binary.exe", io.BytesIO(b"\x00\x01\x02"), "application/octet-stream")},
        )
        assert response.status_code == 400

    async def test_upload_pdf_document(self, client, sample_pdf_bytes):
        """Uploading a valid PDF returns 201."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "report.pdf"
        assert data["file_type"] == "pdf"


class TestListDocuments:
    async def test_list_documents_empty(self, client):
        """GET /api/v1/documents/ returns 200 and an empty list when no docs exist."""
        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_documents_after_upload(self, client, sample_txt_bytes):
        """After uploading one document, the list contains exactly one entry."""
        await client.post(
            "/api/v1/documents/upload",
            files={"file": ("first.txt", io.BytesIO(sample_txt_bytes), "text/plain")},
        )

        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        documents = response.json()
        assert len(documents) == 1
        assert documents[0]["filename"] == "first.txt"

    async def test_list_documents_multiple_uploads(self, client, sample_txt_bytes):
        """After two uploads, the list has two entries."""
        for name in ("alpha.txt", "beta.txt"):
            await client.post(
                "/api/v1/documents/upload",
                files={"file": (name, io.BytesIO(sample_txt_bytes), "text/plain")},
            )

        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        assert len(response.json()) == 2


class TestGetDocument:
    async def test_get_document_not_found(self, client):
        """GET with a random UUID that has no matching document returns 404."""
        missing_id = uuid.uuid4()
        response = await client.get(f"/api/v1/documents/{missing_id}")
        assert response.status_code == 404

    async def test_get_document_after_upload(self, client, sample_txt_bytes):
        """GET with a known document ID returns 200 with document details."""
        upload = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("detail.txt", io.BytesIO(sample_txt_bytes), "text/plain")},
        )
        doc_id = upload.json()["id"]

        response = await client.get(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert data["filename"] == "detail.txt"
        assert "chunks" in data


class TestDeleteDocument:
    async def test_delete_document(self, client, sample_txt_bytes):
        """Upload a document then delete it — expects 204."""
        upload = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("todelete.txt", io.BytesIO(sample_txt_bytes), "text/plain")},
        )
        assert upload.status_code == 201
        doc_id = upload.json()["id"]

        response = await client.delete(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 204

    async def test_delete_document_not_found(self, client):
        """DELETE with an unknown UUID returns 404."""
        missing_id = uuid.uuid4()
        response = await client.delete(f"/api/v1/documents/{missing_id}")
        assert response.status_code == 404

    async def test_delete_removes_from_list(self, client, sample_txt_bytes):
        """After deleting the only document, the list is empty again."""
        upload = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("gone.txt", io.BytesIO(sample_txt_bytes), "text/plain")},
        )
        doc_id = upload.json()["id"]

        await client.delete(f"/api/v1/documents/{doc_id}")

        response = await client.get("/api/v1/documents/")
        assert response.json() == []
