"""Build/rebuild the Chroma vector store from data/docs.

Run from project root:   python -m scripts.ingest
Requires Ollama running with `nomic-embed-text` pulled.
"""

import logging

from app.core.config import CHROMA_DIR, DOCS_DIR
from app.ingestion.document_loader import load_documents_from_folder
from app.rag.vector_store import documents_to_chunks, index_documents

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    documents = load_documents_from_folder(DOCS_DIR)
    print(f"Loaded {len(documents)} documents from {DOCS_DIR}")
    for document in documents:
        n = sum(1 for c in documents_to_chunks([document]))
        print(f"  [{document.id:>2}] {document.category.value:<16} {n} chunk(s)  {document.title}")

    report = index_documents(documents)
    print(f"\nEmbedded {report.chunks_indexed} chunks from {report.documents_indexed} documents")
    print(f"Persisted to {CHROMA_DIR}")


if __name__ == "__main__":
    main()