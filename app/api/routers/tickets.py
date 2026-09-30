from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_knowledge_base_service
from app.api.schemas import (
    CommentCreate, CommentOut, MismatchOut, Page, TicketCreate, TicketOut, TicketUpdate,
)
from app.api.security import require_api_key
from app.models import TicketPriority, TicketStatus
from app.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/tickets", tags=["tickets"], dependencies=[Depends(require_api_key)])


def _get_or_404(service: KnowledgeBaseService, ticket_id: int):
    ticket = service.get_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ticket with id {ticket_id}")
    return ticket


@router.get("", response_model=Page[TicketOut])
def list_tickets(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status_: TicketStatus | None = Query(None, alias="status"),
    priority: TicketPriority | None = Query(None),
    assignee_id: int | None = Query(None, gt=0),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    tickets = service.get_all_tickets(status=status_, priority=priority, assignee_id=assignee_id)
    return Page[TicketOut](
        items=[TicketOut.model_validate(t) for t in tickets[skip: skip + limit]],
        total=len(tickets), skip=skip, limit=limit,
    )


@router.get("/mismatches", response_model=list[MismatchOut])
def list_station_mismatches(service: KnowledgeBaseService = Depends(get_knowledge_base_service)):
    """Station Ownership Mismatch: open tickets assigned to someone outside the station that owns
    the ticket's related document."""
    return [
        MismatchOut(
            ticket_id=m.ticket.id,
            ticket_title=m.ticket.title,
            ticket_priority=m.ticket.priority,
            ticket_status=m.ticket.status,
            related_document_id=m.document.id,
            related_document_title=m.document.title,
            assignee_name=m.assignee.name,
            assignee_station=m.assignee.station,
            document_owner_name=m.owner.name,
            document_owner_station=m.owner.station,
        )
        for m in service.get_station_mismatches()
    ]


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int, service: KnowledgeBaseService = Depends(get_knowledge_base_service)):
    return TicketOut.model_validate(_get_or_404(service, ticket_id))


@router.post("", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(
    body: TicketCreate, service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    return TicketOut.model_validate(service.create_ticket(**body.model_dump()))


@router.patch("/{ticket_id}", response_model=TicketOut)
def update_ticket(
    ticket_id: int,
    body: TicketUpdate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    ticket = service.update_ticket(ticket_id, body.model_dump(exclude_unset=True))
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ticket with id {ticket_id}")
    return TicketOut.model_validate(ticket)


@router.get("/{ticket_id}/comments", response_model=list[CommentOut])
def list_comments(
    ticket_id: int, service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    _get_or_404(service, ticket_id)
    return [CommentOut.model_validate(c) for c in service.get_comments(ticket_id)]


@router.post("/{ticket_id}/comments", response_model=CommentOut, status_code=status.HTTP_201_CREATED)
def add_comment(
    ticket_id: int,
    body: CommentCreate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    comment = service.add_comment(ticket_id, body.author_id, body.body)
    if comment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ticket with id {ticket_id}")
    return CommentOut.model_validate(comment)