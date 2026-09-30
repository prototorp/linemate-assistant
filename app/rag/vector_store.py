"""Ingestion pipeline: Document rows -> chunks -> embeddings -> persisted Chroma store.

The same Document rows that the REST API serves are the source material, there is no separate
copy of the data. Nothing here knows about FastAPI.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import (
    CHROMA_COLLECTION, CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, OLLAMA_URL,
)
from app.models import Document

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


def _metadata(document: Document) -> dict:
    """Chroma metadata may only hold str/int/float/bool, so dates become ISO strings."""
    return {
        "document_id": document.id,
        "title": document.title,
        "category": document.category.value,
        "owner_id": document.owner_id,
        "last_reviewed_at": document.last_reviewed_at.isoformat(),
    }


def documents_to_chunks(documents: list[Document]) -> list[LCDocument]:
    """Split each document body into overlapping chunks that carry the parent's metadata."""
    chunks: list[LCDocument] = []
    for document in documents:
        parent = LCDocument(page_content=document.body, metadata=_metadata(document))
        for index, chunk in enumerate(_splitter.split_documents([parent])):
            chunk.metadata["chunk_index"] = index
            chunks.append(chunk)
    return chunks


@dataclass(frozen=True)
class IngestReport:
    documents_indexed: int
    chunks_indexed: int


def index_documents(
    documents: list[Document],
    persist_directory: str = CHROMA_DIR,
    embeddings=None,
) -> IngestReport:
    """Rebuild the collection from scratch. Safe to run repeatedly: the old collection is
    dropped first and chunk ids are deterministic, so re-running never duplicates chunks."""
    chunks = documents_to_chunks(documents)
    if not chunks:
        raise ValueError("No documents to index")
    embeddings = embeddings or get_embeddings()

    Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    ).delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[f"doc{c.metadata['document_id']}-chunk{c.metadata['chunk_index']}" for c in chunks],
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_directory,  # writes land on disk as they happen
    )
    get_vector_store.cache_clear()  # any cached handle now points at the dropped collection
    return IngestReport(documents_indexed=len(documents), chunks_indexed=len(chunks))


@lru_cache
def get_vector_store(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Reopen the persisted collection without re-chunking or re-embedding anything."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )