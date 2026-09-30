"""/ask routes with a fake AskService: tests the HTTP contract.
The chain itself is tested in test_rag.py."""

import pytest

ROUTES = [
    ("post", "/ask", {"question": "How often do we filter the fryers?"}),
    ("post", "/ask/conversation", {"conversation_id": "c1", "question": "How often?"}),
    ("delete", "/ask/conversation/c1", None),
    ("get", "/ask/retrieve?query=fryer", None),
    ("post", "/ask/reindex", None),
]


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_every_ask_route_requires_api_key(client, method, path, body):
    assert getattr(client, method)(path, **({"json": body} if body else {})).status_code == 401


def test_ask_returns_answer_with_citations(client, auth):
    response = client.post("/ask", json={"question": "How often do we filter the fryers?"}, headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Filter twice a day [1]."
    source = body["sources"][0]
    assert source["title"] == "Fryer Oil Filtration and Change SOP"
    assert source["document_id"] == 2
    assert source["cited_in_answer"] is True
    assert source["is_stale"] is True
    assert body["conversation_id"] is None


def test_ask_passes_k_through(client, auth, fake_ask):
    client.post("/ask", json={"question": "fryer oil?", "k": 7}, headers=auth)
    assert fake_ask.calls[-1] == ("ask", "fryer oil?", 7)


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                          # too short
    {"question": "x" * 1001},                    # too long
    {"question": "fryer oil?", "k": 0},
    {"question": "fryer oil?", "k": 11},
    {},
])
def test_ask_validates_the_request_body(client, auth, payload):
    assert client.post("/ask", json=payload, headers=auth).status_code == 422


def test_conversation_echoes_id_and_rewrites_follow_ups(client, auth):
    first = client.post("/ask/conversation", json={"conversation_id": "c1", "question": "How often do we filter fryers?"}, headers=auth).json()
    second = client.post("/ask/conversation", json={"conversation_id": "c1", "question": "And who owns that SOP?"}, headers=auth).json()
    assert first["conversation_id"] == second["conversation_id"] == "c1"
    assert first["standalone_question"] == "How often do we filter fryers?"
    assert second["standalone_question"].startswith("[rewritten with history]")


def test_conversations_are_independent(client, auth):
    client.post("/ask/conversation", json={"conversation_id": "a", "question": "first question"}, headers=auth)
    other = client.post("/ask/conversation", json={"conversation_id": "b", "question": "first question"}, headers=auth).json()
    assert other["standalone_question"] == "first question"  # b has no history


def test_forget_conversation(client, auth):
    client.post("/ask/conversation", json={"conversation_id": "c1", "question": "first question"}, headers=auth)
    assert client.delete("/ask/conversation/c1", headers=auth).status_code == 204
    assert client.delete("/ask/conversation/c1", headers=auth).status_code == 404


def test_retrieve_returns_chunks_with_staleness(client, auth):
    body = client.get("/ask/retrieve?query=fryer oil&k=2", headers=auth).json()
    assert [c["document_id"] for c in body] == [2, 9]
    assert body[0]["is_stale"] is True       # SOP reviewed 2026-02-10
    assert body[1]["is_stale"] is False      # Incident Reports never go stale
    assert body[0]["content"].startswith("Filter each fryer")


def test_retrieve_can_filter_by_category(client, auth, fake_ask):
    client.get("/ask/retrieve?query=grease fire&category=Incident Report", headers=auth)
    assert fake_ask.calls[-1] == ("retrieve", "grease fire", 4, "Incident Report")


def test_retrieve_validates_query(client, auth):
    assert client.get("/ask/retrieve?query=ab", headers=auth).status_code == 422
    assert client.get("/ask/retrieve", headers=auth).status_code == 422


def test_reindex_reports_counts(client, auth):
    body = client.post("/ask/reindex", headers=auth).json()
    assert body == {"documents_indexed": 5, "chunks_indexed": 10}


def test_llm_backend_down_returns_503_not_a_stack_trace(client, auth, fake_ask):
    fake_ask.fail = True
    response = client.post("/ask", json={"question": "fryer oil?"}, headers=auth)
    assert response.status_code == 503
    assert "LLM" in response.json()["detail"]