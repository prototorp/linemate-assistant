"""Test doubles so no test needs Ollama, Chroma, or the network."""

from langchain_core.documents import Document as LCDocument

from app.core.exceptions import AnswerBackendError
from app.rag.ask_service import AskResult, SourceCitation
from app.rag.vector_store import IngestReport


def make_chunk(
    document_id: int = 2,
    title: str = "Fryer Oil Filtration and Change SOP",
    category: str = "SOP",
    last_reviewed_at: str = "2026-02-10",
    text: str = "Filter each fryer twice per day, at mid-shift and at close.",
    chunk_index: int = 0,
) -> LCDocument:
    return LCDocument(
        page_content=text,
        metadata={
            "document_id": document_id, "title": title, "category": category,
            "owner_id": 102, "last_reviewed_at": last_reviewed_at, "chunk_index": chunk_index,
        },
    )


class FakeRetriever:
    """Returns fixed chunks and remembers every query it was given."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.queries: list[str] = []

    def invoke(self, query: str):
        self.queries.append(query)
        return self.chunks

    def factory(self, k: int = 4):
        return self


class FakeAskService:
    """Stands in for AskService in API tests."""

    def __init__(self):
        self.fail = False
        self.calls: list[tuple] = []
        self._conversations: dict[str, int] = {}

    def _check(self):
        if self.fail:
            raise AnswerBackendError("Could not reach the local LLM.")

    @staticmethod
    def _sources(cited: bool = True):
        return [
            SourceCitation(
                document_id=2, title="Fryer Oil Filtration and Change SOP", category="SOP",
                last_reviewed_at="2026-02-10", is_stale=True,
                excerpt="Filter each fryer twice per day...", cited_in_answer=cited,
                source_numbers=[1],
            )
        ]

    def ask(self, question, k=4):
        self._check()
        self.calls.append(("ask", question, k))
        return AskResult("Filter twice a day [1].", self._sources(), question)

    def ask_in_conversation(self, conversation_id, question, k=4):
        self._check()
        turns = self._conversations.get(conversation_id, 0)
        self._conversations[conversation_id] = turns + 1
        self.calls.append((conversation_id, question, k))
        standalone = question if turns == 0 else f"[rewritten with history] {question}"
        return AskResult(f"answer {turns + 1}", self._sources(), standalone)

    def clear_conversation(self, conversation_id):
        return self._conversations.pop(conversation_id, None) is not None

    def retrieve(self, query, k=4, category=None):
        self._check()
        self.calls.append(("retrieve", query, k, category))
        return [
            make_chunk(),
            make_chunk(
                document_id=9, title="Incident Report: Grease Fire", category="Incident Report",
                last_reviewed_at="2026-03-30", text="Grease built up in the hood.", chunk_index=1,
            ),
        ]

    def reindex(self, documents):
        self._check()
        return IngestReport(documents_indexed=len(documents), chunks_indexed=len(documents) * 2)