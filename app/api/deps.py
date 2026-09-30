"""Dependency-injection providers. Routes ask for these with Depends(...); tests replace them
with app.dependency_overrides, so no test needs real files, the real clock, or Ollama."""

from datetime import date
from functools import lru_cache

from app.core.config import CREW_CSV, DOCS_DIR, TICKETS_CSV
from app.ingestion.crew_loader import load_crew_from_csv
from app.ingestion.document_loader import load_documents_from_folder
from app.ingestion.ticket_loader import load_tickets_from_csv
from app.rag.ask_service import AskService
from app.services.knowledge_base import KnowledgeBaseService


@lru_cache
def get_knowledge_base_service() -> KnowledgeBaseService:
    """Loaded from disk once, then the same instance is reused for every request
    (@lru_cache), so POST/PATCH changes persist for the life of the process."""
    return KnowledgeBaseService(
        documents=load_documents_from_folder(DOCS_DIR),
        tickets=load_tickets_from_csv(TICKETS_CSV),
        crew=load_crew_from_csv(CREW_CSV),
    )


def get_today() -> date:
    """The clock as a dependency, so staleness tests can pin a date."""
    return date.today()


@lru_cache
def get_ask_service() -> AskService:
    return AskService()