"""Shared fixtures. The real app is exercised end to end through TestClient, but with three
dependencies swapped out via app.dependency_overrides:
  * the knowledge base (small deterministic dataset instead of data/*.csv)
  * the clock (pinned to TODAY so staleness is reproducible)
  * the ask service (a fake, so no Ollama is needed)"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ask_service, get_knowledge_base_service, get_today
from app.api.main import app
from app.core.config import API_KEY
from app.services.knowledge_base import KnowledgeBaseService
from tests.factories import TODAY, make_crew, make_documents, make_tickets
from tests.fakes import FakeAskService


@pytest.fixture
def service() -> KnowledgeBaseService:
    return KnowledgeBaseService(make_documents(), make_tickets(), make_crew())


@pytest.fixture
def fake_ask() -> FakeAskService:
    return FakeAskService()


@pytest.fixture
def client(service, fake_ask):
    app.dependency_overrides[get_knowledge_base_service] = lambda: service
    app.dependency_overrides[get_today] = lambda: TODAY
    app.dependency_overrides[get_ask_service] = lambda: fake_ask
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}