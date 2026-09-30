"""TestClient integration tests for /documents: routing, DI, security and validation together"""

def test_requires_api_key(client):
    assert client.get("/documents").status_code == 401


def test_rejects_wrong_api_key(client):
    assert client.get("/documents", headers={"X-API-Key": "nope"}).status_code == 401


def test_health_check_is_public(client):
    assert client.get("/").json() == {"status": "ok", "service": "LineMate"}


def test_list_returns_pagination_envelope(client, auth):
    body = client.get("/documents", headers=auth).json()
    assert (body["total"], body["skip"], body["limit"]) == (5, 0, 10)
    assert len(body["items"]) == 5
    assert "body" not in body["items"][0]  # list view stays light


def test_list_respects_skip_and_limit(client, auth):
    body = client.get("/documents?skip=3&limit=2", headers=auth).json()
    assert [d["id"] for d in body["items"]] == [4, 5]
    assert body["total"] == 5


def test_list_rejects_limit_over_max(client, auth):
    assert client.get("/documents?limit=500", headers=auth).status_code == 422


def test_list_filters_by_category(client, auth):
    body = client.get("/documents?category=Incident Report", headers=auth).json()
    assert [d["id"] for d in body["items"]] == [3]


def test_list_rejects_unknown_category(client, auth):
    assert client.get("/documents?category=Poetry", headers=auth).status_code == 422


def test_get_by_id_includes_body(client, auth):
    response = client.get("/documents/1", headers=auth)
    assert response.status_code == 200
    assert response.json()["title"] == "Steak Guide"
    assert response.json()["body"] == "Rest steaks 5 minutes."


def test_get_by_id_not_found(client, auth):
    assert client.get("/documents/9999", headers=auth).status_code == 404


def test_stale_endpoint_excludes_incident_reports_and_fresh_docs(client, auth):
    body = client.get("/documents/stale", headers=auth).json()
    assert [d["id"] for d in body] == [2, 5]  # most overdue first; d3 is exempt, d4 is exactly 90 days
    assert body[0]["days_since_reviewed"] == 200


def test_stale_threshold_is_a_query_parameter(client, auth):
    body = client.get("/documents/stale?threshold_days=30", headers=auth).json()
    assert [d["id"] for d in body] == [2, 5, 4]


def test_stale_rejects_zero_threshold(client, auth):
    assert client.get("/documents/stale?threshold_days=0", headers=auth).status_code == 422


NEW_DOC = {
    "title": "Sauce Station Setup",
    "category": "SOP",
    "body": "Set up the sauce station with labeled containers.",
    "owner_id": 1,
    "last_reviewed_at": "2026-09-01",
}


def test_create_document_then_fetch_it(client, auth):
    created = client.post("/documents", json=NEW_DOC, headers=auth)
    assert created.status_code == 201
    new_id = created.json()["id"]
    assert new_id == 6
    assert client.get(f"/documents/{new_id}", headers=auth).json()["title"] == "Sauce Station Setup"


def test_create_document_rejects_unknown_owner(client, auth):
    response = client.post("/documents", json={**NEW_DOC, "owner_id": 999}, headers=auth)
    assert response.status_code == 422
    assert "owner_id" in response.json()["detail"]


def test_create_document_rejects_bad_payloads(client, auth):
    assert client.post("/documents", json={**NEW_DOC, "category": "Poetry"}, headers=auth).status_code == 422
    assert client.post("/documents", json={**NEW_DOC, "title": "x"}, headers=auth).status_code == 422
    assert client.post("/documents", json={**NEW_DOC, "last_reviewed_at": "2999-01-01"}, headers=auth).status_code == 422
    missing = {k: v for k, v in NEW_DOC.items() if k != "body"}
    assert client.post("/documents", json=missing, headers=auth).status_code == 422


def test_middleware_adds_tracing_headers(client, auth):
    response = client.get("/documents", headers=auth)
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_crew_endpoint_lists_stations(client, auth):
    body = client.get("/crew", headers=auth).json()
    assert {c["station"] for c in body} == {"Grill", "Prep", "Pastry", "Front of House"}