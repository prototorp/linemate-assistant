"""AskService: what the /ask routes call. Wraps the chain, memory, citations and reindexing.

The routes depend on this class (via Depends), so tests can swap in a fake and never need Ollama.
"""

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date

import httpx
from langchain_core.documents import Document as LCDocument
from langchain_core.runnables import Runnable
from ollama import ResponseError

from app.core.config import DEFAULT_K
from app.core.exceptions import AnswerBackendError
from app.models import Document
from app.rag import qa_chain
from app.rag.retriever import get_similarity_retriever, is_chunk_stale
from app.rag.vector_store import IngestReport, index_documents

MAX_CONVERSATIONS = 200
EXCERPT_CHARS = 240
_BACKEND_ERRORS = (ConnectionError, httpx.ConnectError, httpx.TimeoutException, ResponseError)
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class SourceCitation:
    document_id: int
    title: str
    category: str
    last_reviewed_at: str
    is_stale: bool
    excerpt: str
    cited_in_answer: bool  # True if the answer text references any of this document's [n] labels
    source_numbers: list[int]  # every [n] label that belongs to this document's retrieved chunks


@dataclass
class AskResult:
    answer: str
    sources: list[SourceCitation]
    standalone_question: str


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rstrip() + "..."


def build_citations(
    documents: list[LCDocument], answer: str, as_of: date | None = None
) -> list[SourceCitation]:
    """Citations come from the retrieval metadata, not from trusting the model to name documents.

    The model cites CHUNKS ("[1]", "[2]"), but one document can supply several chunks. So there is
    one entry per source document, and `source_numbers` lists every [n] label that belongs to it.
    That way each number in the answer text maps to exactly one listed document, and
    `cited_in_answer` says whether the answer used any of them."""
    as_of = as_of or date.today()
    cited_numbers = {int(n) for n in _CITATION_RE.findall(answer)}
    by_document: OrderedDict[int, SourceCitation] = OrderedDict()

    for number, chunk in enumerate(documents, start=1):
        meta = chunk.metadata
        document_id = int(meta["document_id"])
        is_cited = number in cited_numbers
        existing = by_document.get(document_id)
        if existing is None:
            by_document[document_id] = SourceCitation(
                document_id=document_id,
                title=meta["title"],
                category=meta["category"],
                last_reviewed_at=meta["last_reviewed_at"],
                is_stale=is_chunk_stale(chunk, as_of),
                excerpt=_excerpt(chunk.page_content),
                cited_in_answer=is_cited,
                source_numbers=[number],
            )
        else:
            existing.source_numbers.append(number)
            if is_cited and not existing.cited_in_answer:
                existing.cited_in_answer = True
                existing.excerpt = _excerpt(chunk.page_content)  # show the chunk the answer used
    return list(by_document.values())


def stale_note(sources: list[SourceCitation], as_of: date | None = None) -> str:
    """A warning for every source the answer actually cited that is overdue for review.

    Written in code, not by the model: a 3B model ignored the instruction to add it. Sources that
    were retrieved but not cited get no note (the answer did not rely on them)."""
    as_of = as_of or date.today()
    sentences = []
    for source in sources:
        if source.cited_in_answer and source.is_stale:
            days = (as_of - date.fromisoformat(source.last_reviewed_at)).days
            sentences.append(
                f'Note: "{source.title}" was last reviewed {days} days ago and may be out of '
                "date, so confirm with your sous chef."
            )
    return " ".join(sentences)


class AskService:
    def __init__(
        self,
        chain: Runnable | None = None,
        summarizer: Runnable | None = None,
    ):
        self._chain = chain or qa_chain.retrieval_chain
        self._summarizer = summarizer or qa_chain.summarization_chain
        self._conversations: OrderedDict[str, qa_chain.ConversationMemory] = OrderedDict()
        self._lock = threading.Lock()

    # ---- conversation memory ------------------------------------------------
    def _memory_for(self, conversation_id: str) -> qa_chain.ConversationMemory:
        with self._lock:
            memory = self._conversations.get(conversation_id)
            if memory is None:
                memory = qa_chain.ConversationMemory()
                self._conversations[conversation_id] = memory
                if len(self._conversations) > MAX_CONVERSATIONS:
                    self._conversations.popitem(last=False)  # drop the oldest
            else:
                self._conversations.move_to_end(conversation_id)
            return memory

    def clear_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    # ---- asking -------------------------------------------------------------
    def _run(
        self, question: str, k: int, memory: qa_chain.ConversationMemory | None
    ) -> AskResult:
        payload = {
            "question": question,
            "k": k,
            "summary": memory.summary if memory else "",
            "recent_turns": list(memory.recent_messages) if memory else [],
        }
        try:
            result = self._chain.invoke(payload)
            if memory is not None:
                qa_chain.record_turn(memory, question, result["answer"], self._summarizer)
        except _BACKEND_ERRORS as exc:
            raise AnswerBackendError(
                "Could not reach the local LLM. Is Ollama running, with llama3.2 and "
                "nomic-embed-text pulled?"
            ) from exc
        sources = build_citations(result["documents"], result["answer"])
        note = stale_note(sources)
        return AskResult(
            answer=f"{result['answer']}\n\n{note}" if note else result["answer"],
            sources=sources,
            standalone_question=result["standalone_question"],
        )

    def ask(self, question: str, k: int = DEFAULT_K) -> AskResult:
        """One-off question with no memory."""
        return self._run(question, k, memory=None)

    def ask_in_conversation(
        self, conversation_id: str, question: str, k: int = DEFAULT_K
    ) -> AskResult:
        return self._run(question, k, memory=self._memory_for(conversation_id))

    # ---- retrieval only (shows the retriever on its own) ----------------------
    def retrieve(self, query: str, k: int = DEFAULT_K, category: str | None = None) -> list[LCDocument]:
        try:
            return get_similarity_retriever(k=k, category=category).invoke(query)
        except _BACKEND_ERRORS as exc:
            raise AnswerBackendError("Could not reach the local embedding model (Ollama).") from exc

    # ---- ingestion ------------------------------------------------------------
    def reindex(self, documents: list[Document]) -> IngestReport:
        try:
            return index_documents(documents)
        except _BACKEND_ERRORS as exc:
            raise AnswerBackendError("Could not reach the local embedding model (Ollama).") from exc