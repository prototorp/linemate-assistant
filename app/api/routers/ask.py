"""Grounded Q&A routes, every answer comes back with citations to the source documents"""

from datetime import date

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_ask_service, get_knowledge_base_service, get_today
from app.api.schemas import (
    AskRequest, AskResponse, ReindexResponse, RetrievedChunkOut, SourceOut,
)
from app.api.security import require_api_key
from app.models import DocumentCategory
from app.rag.ask_service import AskResult, AskService
from app.rag.retriever import is_chunk_stale
from app.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/ask", tags=["ask"], dependencies=[Depends(require_api_key)])


def _to_response(result: AskResult, conversation_id: str | None = None) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        sources=[SourceOut.model_validate(s) for s in result.sources],
        standalone_question=result.standalone_question,
        conversation_id=conversation_id,
    )


@router.post("", response_model=AskResponse, summary="Ask a kitchen operations question")
def ask(request: AskRequest, service: AskService = Depends(get_ask_service)):
    return _to_response(service.ask(request.question, k=request.k))


@router.get("/retrieve", response_model=list[RetrievedChunkOut], summary="Retriever only, no LLM")
def retrieve_chunks(
    query: str = Query(min_length=3, description="Natural-language query"),
    k: int = Query(4, ge=1, le=10),
    category: DocumentCategory | None = Query(None),
    service: AskService = Depends(get_ask_service),
    today: date = Depends(get_today),
):
    chunks = service.retrieve(query, k=k, category=category.value if category else None)
    return [
        RetrievedChunkOut(
            document_id=int(c.metadata["document_id"]),
            title=c.metadata["title"],
            category=c.metadata["category"],
            chunk_index=int(c.metadata.get("chunk_index", 0)),
            last_reviewed_at=c.metadata["last_reviewed_at"],
            is_stale=is_chunk_stale(c, today),
            content=c.page_content,
        )
        for c in chunks
    ]


@router.post("/reindex", response_model=ReindexResponse, summary="Re-chunk and re-embed all documents")
def reindex(
    ask_service: AskService = Depends(get_ask_service),
    kb: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    report = ask_service.reindex(kb.get_all_documents())
    return ReindexResponse(
        documents_indexed=report.documents_indexed, chunks_indexed=report.chunks_indexed
    )