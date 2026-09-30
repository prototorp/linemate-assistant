from datetime import date

from app.models import Document, DocumentCategory
from app.rag.vector_store import documents_to_chunks


def test_chunks_carry_parent_metadata_and_index():
    body = " ".join(f"Sentence number {i} about fryers." for i in range(60))  # long enough to split
    doc = Document(7, "Long Doc", DocumentCategory.SOP, body, 1, date(2026, 1, 1))
    chunks = documents_to_chunks([doc])
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["document_id"] == 7 and c.metadata["category"] == "SOP" for c in chunks)
    assert all(c.metadata["last_reviewed_at"] == "2026-01-01" for c in chunks)  # ISO string, Chroma-safe
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_short_document_is_a_single_chunk():
    doc = Document(1, "Short", DocumentCategory.RECIPE, "Bake it.", 1, date(2026, 1, 1))
    assert len(documents_to_chunks([doc])) == 1