from datetime import date

from langchain_core.documents import Document as LCDocument

from app.core import rules
from app.core.config import DEFAULT_K
from app.rag.vector_store import get_vector_store


def get_similarity_retriever(k: int = DEFAULT_K, category: str | None = None):
    """Plain nearest-neighbour retrieval, optionally limited to one document category."""
    search_kwargs: dict = {"k": k}
    if category is not None:
        search_kwargs["filter"] = {"category": category}
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def get_diverse_retriever(k: int = DEFAULT_K, fetch_k: int = 20, lambda_mult: float = 0.5):
    """MMR retrieval: fetch `fetch_k` candidates, then re-rank for relevance AND variety so the
    results are not near-duplicates of one another."""
    return get_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def is_chunk_stale(document: LCDocument, as_of: date) -> bool:
    meta = document.metadata
    return rules.is_stale(meta["category"], date.fromisoformat(meta["last_reviewed_at"]), as_of)


def format_retrieved_context(documents: list[LCDocument], as_of: date | None = None) -> str:
    """Numbered source blocks. The number is what the model cites, e.g. "[2]".

    Staleness is computed here in code (not left to the model) and written into the header,
    so an out-of-date SOP is visibly flagged whenever it is used as evidence."""
    as_of = as_of or date.today()
    blocks = []
    for number, document in enumerate(documents, start=1):
        meta = document.metadata
        header = f"[{number}] {meta['title']} ({meta['category']}, last reviewed {meta['last_reviewed_at']}"
        if is_chunk_stale(document, as_of):
            days = (as_of - date.fromisoformat(meta["last_reviewed_at"])).days
            header += f"; STALE: not reviewed in {days} days"
        blocks.append(f"{header})\n{document.page_content}")
    return "\n\n".join(blocks)