import logging
 
from app.ingestion.ticket_loader import load_tickets_from_csv
from app.models import TicketPriority, TicketStatus
 
HEADER = "id,title,priority,status,assignee_id,related_document_id,created_at\n"
 
 
def _csv(tmp_path, rows):
    path = tmp_path / "tickets.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path
 
 
def test_loads_valid_row(tmp_path):
    tickets = load_tickets_from_csv(_csv(tmp_path, ["1,Fryer down,High,Open,101,2,2026-09-20T09:15:00"]))
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.priority == TicketPriority.HIGH
    assert ticket.status == TicketStatus.OPEN
    assert ticket.related_document_id == 2
    assert ticket.created_at.day == 20
 
 
def test_blank_related_document_becomes_none(tmp_path):
    tickets = load_tickets_from_csv(_csv(tmp_path, ["1,Drain slow,Low,Open,101,,2026-09-20T09:15:00"]))
    assert tickets[0].related_document_id is None
 
 
def test_skips_invalid_priority_and_status(tmp_path, caplog):
    path = _csv(tmp_path, [
        "1,Missing priority,,Open,101,1,2026-09-20T09:15:00",
        "2,Bad status,Low,Blocked,101,1,2026-09-20T09:15:00",
        "3,Good one,Low,Open,101,1,2026-09-20T09:15:00",
    ])
    with caplog.at_level(logging.WARNING):
        tickets = load_tickets_from_csv(path)
    assert [t.id for t in tickets] == [3]
    assert "invalid priority" in caplog.text
    assert "invalid status" in caplog.text
 
 
def test_skips_non_integer_assignee(tmp_path):
    assert load_tickets_from_csv(_csv(tmp_path, ["1,Bad,Low,Open,abc,1,2026-09-20T09:15:00"])) == []
 
 
def test_skips_duplicate_ids(tmp_path):
    rows = ["1,First,Low,Open,101,1,2026-09-20T09:15:00", "1,Again,Low,Open,101,1,2026-09-20T09:15:00"]
    assert [t.title for t in load_tickets_from_csv(_csv(tmp_path, rows))] == ["First"]
 
 
def test_skips_invalid_created_at(tmp_path):
    assert load_tickets_from_csv(_csv(tmp_path, ["1,Bad date,Low,Open,101,1,yesterday"])) == []