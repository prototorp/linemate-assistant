from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_knowledge_base_service, get_today
from app.api.schemas import DocumentCreate, DocumentDetail, DocumentOut, Page, StaleDocumentOut
from app.api.security import require_api_key
from app.core.config import STALE_THRESHOLD_DAYS
from app.models import DocumentCategory
from app.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=Page[DocumentOut])
def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max documents to return"),
    category: DocumentCategory | None = Query(None, description="Filter by category"),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    documents = service.get_all_documents()
    if category is not None:
        documents = [d for d in documents if d.category == category]
    return Page[DocumentOut](
        items=[DocumentOut.model_validate(d) for d in documents[skip: skip + limit]],
        total=len(documents), skip=skip, limit=limit,
    )


@router.get("/stale", response_model=list[StaleDocumentOut])
def list_stale_documents(
    threshold_days: int = Query(STALE_THRESHOLD_DAYS, ge=1, description="Days without review"),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    today: date = Depends(get_today),
):
    """Stale Documentation: non-Incident-Report documents not reviewed within `threshold_days`."""
    return [
        StaleDocumentOut(
            **DocumentOut.model_validate(item.document).model_dump(),
            days_since_reviewed=item.days_since_reviewed,
        )
        for item in service.get_stale_documents(today, threshold_days)
    ]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: int, service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    document = service.get_document_by_id(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document with id {document_id}")
    return DocumentDetail.model_validate(document)


@router.post("", response_model=DocumentDetail, status_code=status.HTTP_201_CREATED)
def create_document(
    body: DocumentCreate, service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """Adds a document to the in-memory store. Call POST /ask/reindex afterwards so /ask can see it."""
    return DocumentDetail.model_validate(service.create_document(**body.model_dump()))