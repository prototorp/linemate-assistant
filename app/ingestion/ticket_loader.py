"""Loads tickets.csv, bad rows are logged and skipped."""

import csv
import logging
from datetime import datetime
from pathlib import Path

from app.core.exceptions import TicketLoadError
from app.models import Ticket, TicketPriority, TicketStatus

log = logging.getLogger("linemate.ingestion")


def _row_to_ticket(row: dict[str, str]) -> Ticket:
    row_id = row.get("id")
    try:
        priority = TicketPriority(row["priority"])
    except (ValueError, KeyError) as exc:
        raise TicketLoadError(f"ticket row {row_id}: invalid priority {row.get('priority')!r}") from exc

    try:
        status = TicketStatus(row["status"])
    except (ValueError, KeyError) as exc:
        raise TicketLoadError(f"ticket row {row_id}: invalid status {row.get('status')!r}") from exc

    try:
        ticket_id = int(row["id"])
        assignee_id = int(row["assignee_id"])
        related = row.get("related_document_id") or ""
        related_document_id = int(related) if related.strip() else None  # blank is allowed
    except (ValueError, KeyError, TypeError) as exc:
        raise TicketLoadError(
            f"ticket row {row_id}: id, assignee_id and related_document_id must be integers"
        ) from exc

    created_raw = (row.get("created_at") or "").strip()
    try:
        created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.now()
    except ValueError as exc:
        raise TicketLoadError(f"ticket row {row_id}: invalid created_at {created_raw!r}") from exc

    title = (row.get("title") or "").strip()
    if not title:
        raise TicketLoadError(f"ticket row {row_id}: title is empty")

    return Ticket(
        id=ticket_id, title=title, priority=priority, assignee_id=assignee_id,
        status=status, related_document_id=related_document_id, created_at=created_at,
    )


def load_tickets_from_csv(csv_path: str | Path) -> list[Ticket]:
    tickets: list[Ticket] = []
    seen_ids: set[int] = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:  # newline="" is required by csv
        for row in csv.DictReader(handle):
            try:
                ticket = _row_to_ticket(row)
                if ticket.id in seen_ids:
                    raise TicketLoadError(f"ticket row {ticket.id}: duplicate id")
            except TicketLoadError as exc:
                log.warning("SKIPPED %s", exc)
                continue
            seen_ids.add(ticket.id)
            tickets.append(ticket)
    return tickets