from datetime import date

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_today
from app.api.schemas import RetrievedChunkOut
from app.api.security import require_api_key
from app.models import DocumentCategory
from app.rag.retriever import get_similarity_retriever, is_chunk_stale

router = APIRouter(prefix="/ask", tags=["ask"], dependencies=[Depends(require_api_key)])


@router.get("/retrieve", response_model=list[RetrievedChunkOut], summary="Retriever only, no LLM")
def retrieve_chunks(
    query: str = Query(min_length=3, description="Natural-language query"),
    k: int = Query(4, ge=1, le=10),
    category: DocumentCategory | None = Query(None),
    today: date = Depends(get_today),
):
    try:
        chunks = get_similarity_retriever(
            k=k, category=category.value if category else None
        ).invoke(query)
    except (ConnectionError, httpx.ConnectError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not reach the local embedding model. Is Ollama running?",
        ) from exc
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