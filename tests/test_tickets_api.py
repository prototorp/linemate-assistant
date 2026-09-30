def test_requires_api_key(client):
    assert client.get("/tickets").status_code == 401


def test_list_returns_pagination_envelope(client, auth):
    body = client.get("/tickets", headers=auth).json()
    assert (body["total"], body["skip"], body["limit"]) == (8, 0, 10)


def test_list_respects_limit_and_rejects_over_max(client, auth):
    assert len(client.get("/tickets?limit=2", headers=auth).json()["items"]) == 2
    assert client.get("/tickets?limit=500", headers=auth).status_code == 422


def test_filter_by_status(client, auth):
    body = client.get("/tickets?status=Open", headers=auth).json()
    assert sorted(t["id"] for t in body["items"]) == [1, 2, 5, 6, 8]


def test_filter_by_priority_and_assignee(client, auth):
    body = client.get("/tickets?priority=High&assignee_id=1", headers=auth).json()
    assert [t["id"] for t in body["items"]] == [2]


def test_filter_rejects_unknown_status(client, auth):
    assert client.get("/tickets?status=Blocked", headers=auth).status_code == 422


def test_get_by_id(client, auth):
    body = client.get("/tickets/2", headers=auth).json()
    assert body["title"] == "Cooler log gaps"
    assert body["related_document_id"] == 2


def test_get_by_id_not_found(client, auth):
    assert client.get("/tickets/9999", headers=auth).status_code == 404


def test_mismatches_endpoint(client, auth):
    body = client.get("/tickets/mismatches", headers=auth).json()
    assert sorted(m["ticket_id"] for m in body) == [2, 3]
    first = next(m for m in body if m["ticket_id"] == 2)
    assert first["assignee_station"] == "Grill"
    assert first["document_owner_station"] == "Prep"
    assert first["related_document_title"] == "Cooler SOP"


NEW_TICKET = {"title": "Sauce cooler too warm", "priority": "High", "assignee_id": 3, "related_document_id": 2}


def test_create_ticket_defaults_to_open(client, auth):
    response = client.post("/tickets", json=NEW_TICKET, headers=auth)
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 9 and body["status"] == "Open"


def test_create_ticket_rejects_unknown_references(client, auth):
    assert client.post("/tickets", json={**NEW_TICKET, "assignee_id": 999}, headers=auth).status_code == 422
    assert client.post("/tickets", json={**NEW_TICKET, "related_document_id": 999}, headers=auth).status_code == 422


def test_create_ticket_rejects_bad_payloads(client, auth):
    assert client.post("/tickets", json={**NEW_TICKET, "priority": "Urgent"}, headers=auth).status_code == 422
    assert client.post("/tickets", json={**NEW_TICKET, "title": ""}, headers=auth).status_code == 422
    assert client.post("/tickets", json={"title": "No assignee", "priority": "Low"}, headers=auth).status_code == 422


def test_patch_changes_only_the_fields_sent(client, auth):
    response = client.patch("/tickets/2", json={"status": "In-Progress"}, headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "In-Progress"
    assert body["priority"] == "High" and body["assignee_id"] == 1  # untouched


def test_resolving_a_ticket_removes_it_from_mismatches(client, auth):
    client.patch("/tickets/2", json={"status": "Resolved"}, headers=auth)
    body = client.get("/tickets/mismatches", headers=auth).json()
    assert [m["ticket_id"] for m in body] == [3]


def test_reassigning_to_the_owning_station_clears_the_mismatch(client, auth):
    client.patch("/tickets/2", json={"assignee_id": 3}, headers=auth)  # 3 is on Prep
    assert [m["ticket_id"] for m in client.get("/tickets/mismatches", headers=auth).json()] == [3]


def test_patch_can_unlink_the_document(client, auth):
    body = client.patch("/tickets/2", json={"related_document_id": None}, headers=auth).json()
    assert body["related_document_id"] is None


def test_patch_rejects_nulls_and_unknown_refs(client, auth):
    assert client.patch("/tickets/2", json={"status": None}, headers=auth).status_code == 422
    assert client.patch("/tickets/2", json={"assignee_id": 999}, headers=auth).status_code == 422
    assert client.patch("/tickets/2", json={"status": "Blocked"}, headers=auth).status_code == 422


def test_patch_not_found(client, auth):
    assert client.patch("/tickets/9999", json={"status": "Closed"}, headers=auth).status_code == 404


def test_comments_round_trip(client, auth):
    assert client.get("/tickets/2/comments", headers=auth).json() == []
    created = client.post("/tickets/2/comments", json={"author_id": 3, "body": "Checked the log."}, headers=auth)
    assert created.status_code == 201
    listed = client.get("/tickets/2/comments", headers=auth).json()
    assert [c["body"] for c in listed] == ["Checked the log."]


def test_comment_validation(client, auth):
    assert client.post("/tickets/9999/comments", json={"author_id": 3, "body": "x"}, headers=auth).status_code == 404
    assert client.post("/tickets/2/comments", json={"author_id": 999, "body": "x"}, headers=auth).status_code == 422
    assert client.post("/tickets/2/comments", json={"author_id": 3, "body": ""}, headers=auth).status_code == 422
    assert client.get("/tickets/9999/comments", headers=auth).status_code == 404