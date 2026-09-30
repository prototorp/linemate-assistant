"""Pydantic v2 request/response models for the API.

Deliberately separate from app.models (plain dataclasses): these describe only what the API is
willing to accept and show. `from_attributes=True` lets Model.model_validate(obj) read plain objects.
"""

from datetime import date, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models import DocumentCategory, Station, TicketPriority, TicketStatus

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """Pagination envelope: the page of results plus enough metadata to ask for the next one."""
    items: list[T]
    total: int
    skip: int
    limit: int


# ---- crew ------------------------------------------------------------------
class CrewMemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    station: Station


# ---- documents ---------------------------------------------------------------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    category: DocumentCategory
    owner_id: int
    last_reviewed_at: date


class DocumentDetail(DocumentOut):
    body: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    category: DocumentCategory
    body: str = Field(min_length=10)
    owner_id: int = Field(gt=0)
    last_reviewed_at: date

    @model_validator(mode="after")
    def not_in_the_future(self):
        if self.last_reviewed_at > date.today():
            raise ValueError("last_reviewed_at cannot be in the future")
        return self


class StaleDocumentOut(DocumentOut):
    days_since_reviewed: int


# ---- tickets -----------------------------------------------------------------
class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    priority: TicketPriority
    status: TicketStatus
    assignee_id: int
    related_document_id: int | None
    created_at: datetime


class TicketCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    priority: TicketPriority
    assignee_id: int = Field(gt=0)
    related_document_id: int | None = Field(default=None, gt=0)


class TicketUpdate(BaseModel):
    """PATCH body: only the fields the client sends are changed."""
    title: str | None = Field(default=None, min_length=3, max_length=200)
    priority: TicketPriority | None = None
    status: TicketStatus | None = None
    assignee_id: int | None = Field(default=None, gt=0)
    related_document_id: int | None = Field(default=None, gt=0)  # send null to unlink

    @model_validator(mode="after")
    def no_null_for_required_fields(self):
        for name in ("title", "priority", "status", "assignee_id"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class MismatchOut(BaseModel):
    ticket_id: int
    ticket_title: str
    ticket_priority: TicketPriority
    ticket_status: TicketStatus
    related_document_id: int
    related_document_title: str
    assignee_name: str
    assignee_station: Station
    document_owner_name: str
    document_owner_station: Station


class CommentCreate(BaseModel):
    author_id: int = Field(gt=0)
    body: str = Field(min_length=1, max_length=2000)


class CommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ticket_id: int
    author_id: int
    body: str
    created_at: datetime


# ---- analytics ---------------------------------------------------------------
class StationWorkload(BaseModel):
    station: Station
    open_ticket_count: int
    by_priority: dict[str, int]
    load_score: float
    ticket_share_pct: float
    load_share_pct: float
    is_overloaded: bool


class WorkloadReport(BaseModel):
    stations: list[StationWorkload]
    total_open_tickets: int
    unknown_station_open_tickets: int
    mean_load_score: float
    std_load_score: float
    overload_rule: str


# ---- ask ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How often do we filter the fryers?"])
    k: int = Field(default=4, ge=1, le=10, description="How many chunks to retrieve")


class ConversationAskRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64, examples=["shift-2026-09-28"])
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    document_id: int
    title: str
    category: str
    last_reviewed_at: str
    is_stale: bool
    excerpt: str
    cited_in_answer: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    standalone_question: str = Field(description="The question as actually used for retrieval")
    conversation_id: str | None = None


class RetrievedChunkOut(BaseModel):
    document_id: int
    title: str
    category: str
    chunk_index: int
    last_reviewed_at: str
    is_stale: bool
    content: str


class ReindexResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int