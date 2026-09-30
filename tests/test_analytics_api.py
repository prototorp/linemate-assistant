def test_requires_api_key(client):
    assert client.get("/analytics").status_code == 401


def test_workload_report_shape_and_values(client, auth):
    body = client.get("/analytics", headers=auth).json()
    assert body["total_open_tickets"] == 6
    assert [s["station"] for s in body["stations"]] == ["Grill", "Pastry", "Prep", "Front of House"]
    grill = body["stations"][0]
    assert grill["open_ticket_count"] == 4
    assert grill["by_priority"] == {"Low": 2, "Medium": 1, "High": 1, "Critical": 0}
    assert grill["is_overloaded"] is True
    assert sum(1 for s in body["stations"] if s["is_overloaded"]) == 1


def test_workload_alias_returns_the_same_report(client, auth):
    assert client.get("/analytics/workload", headers=auth).json() == client.get("/analytics", headers=auth).json()


def test_workload_reacts_to_ticket_changes(client, auth):
    for ticket_id in (1, 2, 5, 6):  # resolve every open Grill ticket
        client.patch(f"/tickets/{ticket_id}", json={"status": "Resolved"}, headers=auth)
    body = client.get("/analytics", headers=auth).json()
    assert body["stations"][0]["open_ticket_count"] == 0
    assert body["total_open_tickets"] == 2