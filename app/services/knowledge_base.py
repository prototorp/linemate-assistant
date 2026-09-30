"""KnowledgeBaseService: owns the in-memory Document / Ticket / Comment / CrewMember data.

No database on purpose. State lives in plain lists inside one
service object, which FastAPI hands to routes through dependency injection. The heavy
analysis lives in app.analytics; this class only stores data and delegates.
"""

from datetime import date, datetime

from app.analytics.queries import (
    StaleDocument,
    StationMismatch,
    find_stale_documents,
    find_station_mismatches,
)
from app.analytics.workload import compute_station_workload
from app.core.config import STALE_THRESHOLD_DAYS
from app.core.exceptions import InvalidReferenceError
from app.models import (
    Comment, CrewMember, Document, DocumentCategory, Ticket, TicketPriority, TicketStatus,
)


class KnowledgeBaseService:
    def __init__(
        self,
        documents: list[Document],
        tickets: list[Ticket],
        crew: list[CrewMember],
        comments: list[Comment] | None = None,
    ):
        self._documents = list(documents)
        self._tickets = list(tickets)
        self._crew = list(crew)
        self._comments = list(comments or [])

    # ---- crew -------------------------------------------------------------
    def get_all_crew(self) -> list[CrewMember]:
        return self._crew

    def get_crew_member(self, crew_id: int) -> CrewMember | None:
        return next((c for c in self._crew if c.id == crew_id), None)

    # ---- documents --------------------------------------------------------
    def get_all_documents(self) -> list[Document]:
        return self._documents

    def get_document_by_id(self, document_id: int) -> Document | None:
        return next((d for d in self._documents if d.id == document_id), None)

    def create_document(
        self, title: str, category: DocumentCategory, body: str, owner_id: int,
        last_reviewed_at: date,
    ) -> Document:
        if self.get_crew_member(owner_id) is None:
            raise InvalidReferenceError(f"owner_id {owner_id} is not a crew member")
        document = Document(
            id=max((d.id for d in self._documents), default=0) + 1,
            title=title, category=category, body=body,
            owner_id=owner_id, last_reviewed_at=last_reviewed_at,
        )
        self._documents.append(document)
        return document

    def get_stale_documents(
        self, as_of: date, threshold_days: int = STALE_THRESHOLD_DAYS
    ) -> list[StaleDocument]:
        return find_stale_documents(self._documents, as_of, threshold_days)

    # ---- tickets ----------------------------------------------------------
    def get_all_tickets(
        self,
        status: TicketStatus | None = None,
        priority: TicketPriority | None = None,
        assignee_id: int | None = None,
    ) -> list[Ticket]:
        return [
            t for t in self._tickets
            if (status is None or t.status == status)
            and (priority is None or t.priority == priority)
            and (assignee_id is None or t.assignee_id == assignee_id)
        ]

    def get_ticket_by_id(self, ticket_id: int) -> Ticket | None:
        return next((t for t in self._tickets if t.id == ticket_id), None)

    def _check_ticket_refs(self, assignee_id: int | None, related_document_id: int | None) -> None:
        if assignee_id is not None and self.get_crew_member(assignee_id) is None:
            raise InvalidReferenceError(f"assignee_id {assignee_id} is not a crew member")
        if related_document_id is not None and self.get_document_by_id(related_document_id) is None:
            raise InvalidReferenceError(f"related_document_id {related_document_id} does not exist")

    def create_ticket(
        self, title: str, priority: TicketPriority, assignee_id: int,
        related_document_id: int | None = None,
    ) -> Ticket:
        self._check_ticket_refs(assignee_id, related_document_id)
        ticket = Ticket(
            id=max((t.id for t in self._tickets), default=0) + 1,
            title=title, priority=priority, assignee_id=assignee_id,
            status=TicketStatus.OPEN, related_document_id=related_document_id,
            created_at=datetime.now(),
        )
        self._tickets.append(ticket)
        return ticket

    def update_ticket(self, ticket_id: int, changes: dict) -> Ticket | None:
        """Apply only the fields present in `changes` (PATCH semantics)."""
        ticket = self.get_ticket_by_id(ticket_id)
        if ticket is None:
            return None
        self._check_ticket_refs(changes.get("assignee_id"), changes.get("related_document_id"))
        for name, value in changes.items():
            setattr(ticket, name, value)
        return ticket

    def get_station_mismatches(self) -> list[StationMismatch]:
        return find_station_mismatches(self._tickets, self._documents, self._crew)

    # ---- comments ---------------------------------------------------------
    def get_comments(self, ticket_id: int) -> list[Comment]:
        return [c for c in self._comments if c.ticket_id == ticket_id]

    def add_comment(self, ticket_id: int, author_id: int, body: str) -> Comment | None:
        if self.get_ticket_by_id(ticket_id) is None:
            return None
        if self.get_crew_member(author_id) is None:
            raise InvalidReferenceError(f"author_id {author_id} is not a crew member")
        comment = Comment(
            id=max((c.id for c in self._comments), default=0) + 1,
            ticket_id=ticket_id, author_id=author_id, body=body,
        )
        self._comments.append(comment)
        return comment

    # ---- analytics --------------------------------------------------------
    def get_station_workload_report(self) -> dict:
        return compute_station_workload(self._tickets, self._crew)