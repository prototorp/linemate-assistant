"""Retriever helpers: citation-ready context and staleness flags. No Ollama or Chroma needed."""

from datetime import date

from langchain_core.documents import Document as LCDocument

from app.rag.retriever import format_retrieved_context, is_chunk_stale

AS_OF = date(2026, 9, 28)


def chunk(category="SOP", reviewed="2026-02-10", title="Fryer Oil Filtration and Change SOP"):
    return LCDocument(
        page_content="Filter each fryer twice per day.",
        metadata={"document_id": 2, "title": title, "category": category, "last_reviewed_at": reviewed},
    )


def test_context_is_numbered_and_flags_stale_sources():
    context = format_retrieved_context(
        [chunk(), chunk(title="Steak Guide", reviewed="2026-09-02")], as_of=AS_OF
    )
    assert context.startswith("[1] Fryer Oil Filtration and Change SOP")
    assert "STALE: not reviewed in 230 days" in context
    assert "[2] Steak Guide" in context
    assert context.count("STALE") == 1


def test_incident_reports_are_never_flagged_stale():
    old_incident = chunk(category="Incident Report", reviewed="2020-01-01")
    assert not is_chunk_stale(old_incident, AS_OF)
    assert "STALE" not in format_retrieved_context([old_incident], as_of=AS_OF)


def test_recent_document_is_not_stale():
    assert not is_chunk_stale(chunk(reviewed="2026-09-02"), AS_OF)