from dataclasses import dataclass, field
from datetime import datetime

from app.models.enums import TicketPriority, TicketStatus


@dataclass
class Ticket:
    id: int
    title: str
    priority: TicketPriority
    assignee_id: int
    status: TicketStatus = TicketStatus.OPEN
    related_document_id: int | None = None
    created_at: datetime = field(default_factory=datetime.now)