"""add office file types to documents.file_type constraint

Revision ID: 002
Revises: 001
Create Date: 2026-04-03 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_document_file_type", "documents", type_="check")
    op.create_check_constraint(
        "ck_document_file_type",
        "documents",
        "file_type IN ('pdf', 'txt', 'docx', 'xlsx', 'pptx')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_document_file_type", "documents", type_="check")
    op.create_check_constraint(
        "ck_document_file_type",
        "documents",
        "file_type IN ('pdf', 'txt')",
    )
