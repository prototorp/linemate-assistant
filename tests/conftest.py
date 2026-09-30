"""Shared fixtures. The real app is exercised end to end through TestClient, but with two
dependencies swapped out via app.dependency_overrides:
  * the knowledge base (small deterministic dataset instead of data/*.csv)
  * the clock (pinned to TODAY so staleness is reproducible)"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_knowledge_base_service, get_today
from app.api.main import app
from app.core.config import API_KEY
from app.services.knowledge_base import KnowledgeBaseService
from tests.factories import TODAY, make_crew, make_documents, make_tickets


@pytest.fixture
def service() -> KnowledgeBaseService:
    return KnowledgeBaseService(make_documents(), make_tickets(), make_crew())


@pytest.fixture
def client(service):
    app.dependency_overrides[get_knowledge_base_service] = lambda: service
    app.dependency_overrides[get_today] = lambda: TODAY
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}