"""Business-logic queries

  * Stale Documentation        -> find_stale_documents
  * Station Ownership Mismatch -> find_station_mismatches
"""

from dataclasses import dataclass
from datetime import date

from app.core.config import OPEN_STATUSES, STALE_THRESHOLD_DAYS
from app.models import CrewMember, Document, Ticket


@dataclass(frozen=True)
class StaleDocument:
    document: Document
    days_since_reviewed: int


@dataclass(frozen=True)
class StationMismatch:
    ticket: Ticket
    document: Document
    assignee: CrewMember
    owner: CrewMember


def find_stale_documents(
    documents: list[Document], as_of: date, threshold_days: int = STALE_THRESHOLD_DAYS
) -> list[StaleDocument]:
    """Non-Incident-Report documents not reviewed within `threshold_days`, most overdue first."""
    stale = [
        StaleDocument(doc, doc.days_since_reviewed(as_of))
        for doc in documents
        if doc.is_stale(as_of, threshold_days)
    ]
    return sorted(stale, key=lambda item: item.days_since_reviewed, reverse=True)


def find_station_mismatches(
    tickets: list[Ticket], documents: list[Document], crew: list[CrewMember]
) -> list[StationMismatch]:
    """OPEN tickets whose assignee works a different station than the station that owns the
    related document. "Open" means status Open or In-Progress (see OPEN_STATUSES)."""
    documents_by_id = {d.id: d for d in documents}
    crew_by_id = {c.id: c for c in crew}

    mismatches: list[StationMismatch] = []
    for ticket in tickets:
        if ticket.status.value not in OPEN_STATUSES or ticket.related_document_id is None:
            continue
        document = documents_by_id.get(ticket.related_document_id)
        assignee = crew_by_id.get(ticket.assignee_id)
        owner = crew_by_id.get(document.owner_id) if document else None
        if document is None or assignee is None or owner is None:
            continue  # dangling reference: cannot compare stations
        if assignee.station != owner.station:
            mismatches.append(StationMismatch(ticket, document, assignee, owner))
    return mismatches