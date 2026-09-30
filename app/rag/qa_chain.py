"""The formal LCEL retrieval chain and conversation memory

retrieval_chain is ONE composed Runnable. Each RunnablePassthrough.assign(...) adds a key to the
running dict, so every later step can see what the earlier steps produced:

    {question, k, summary, recent_turns}
      -> standalone_question   (follow-ups are rewritten; first questions pass straight through)
      -> documents             (retriever over the persisted Chroma store)
      -> context               (numbered, citation-ready source blocks)
      -> answer                (grounded LLM answer, or a fixed "not found" reply if nothing retrieved)
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from app.core.config import DEFAULT_K, LLM_MODEL, MAX_RECENT_MESSAGES, OLLAMA_URL
from app.rag.retriever import format_retrieved_context, get_similarity_retriever

NO_ANSWER = (
    "I couldn't find anything about that in Hearthline's documents. "
    "Try rephrasing the question, or check with your sous chef."
)


@lru_cache
def get_llm() -> ChatOllama:
    # temperature 0: for grounded Q&A we want the same answer every time, not creativity
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.0)


_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are LineMate, the internal kitchen operations assistant for Hearthline restaurants. "
        "Answer the crew member's question using ONLY the numbered sources below. Do not use "
        "outside knowledge and do not invent details that are not in the sources. "
        "Cite the sources you used inline as [1], [2], and so on. "
        "If a source is marked STALE, tell the crew member the guidance may be out of date. "
        "If the sources do not contain the answer, say so plainly instead of guessing. "
        "Be short and practical: the crew is busy.\n\n"
        "Sources:\n{context}\n\n"
        "Summary of the conversation so far (empty if this is the first question): {summary}",
    ),
    MessagesPlaceholder("recent_turns"),
    ("human", "{question}"),
])

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite follow-up questions. Using the conversation below, rewrite the user's latest "
        "question as one complete standalone question that makes sense without the conversation "
        "(replace pronouns like 'it' or 'that' with what they refer to). If the question is "
        "already standalone, repeat it unchanged. Output ONLY the question, nothing else.\n\n"
        "Summary of earlier conversation (may be empty): {summary}",
    ),
    MessagesPlaceholder("recent_turns"),
    ("human", "Latest question to rewrite: {question}"),
])

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the conversation turns below in 2-3 sentences, keeping every fact a crew member "
        "would need to answer a follow-up question (station, equipment, document names). "
        "If an existing summary is given, build on it instead of starting over. Write only the "
        "summary itself.\n\nExisting summary (empty if none): {existing_summary}",
    ),
    MessagesPlaceholder("turns_to_summarize"),
    ("human", "Summarize the conversation above."),
])


def build_retrieval_chain(
    llm=None, retriever_factory: Callable[..., object] | None = None
) -> Runnable:
    """Compose the chain. `llm` and `retriever_factory` are injectable so tests can pass fakes."""
    llm = llm or get_llm()
    retriever_factory = retriever_factory or get_similarity_retriever

    condense_chain = (
        _CONDENSE_PROMPT | llm | StrOutputParser()
        | RunnableLambda(lambda text: text.strip().strip('"'))
    )
    answer_chain = _ANSWER_PROMPT | llm | StrOutputParser()

    # Only pay for a rewrite call when there is history to resolve.
    standalone_question = RunnableBranch(
        (lambda x: bool(x.get("summary") or x.get("recent_turns")), condense_chain),
        RunnableLambda(lambda x: x["question"]),
    )

    def retrieve(x: dict) -> list:
        # Rebuilt per call so a re-index is picked up immediately.
        return retriever_factory(k=x.get("k", DEFAULT_K)).invoke(x["standalone_question"])

    # Nothing retrieved -> fixed reply, no LLM call, nothing to hallucinate from.
    answer_step = RunnableBranch(
        (lambda x: not x["documents"], RunnableLambda(lambda x: NO_ANSWER)),
        answer_chain,
    )

    return (
        RunnablePassthrough.assign(standalone_question=standalone_question)
        | RunnablePassthrough.assign(documents=RunnableLambda(retrieve))
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_retrieved_context(x["documents"]))
        )
        | RunnablePassthrough.assign(answer=answer_step)
    )


def build_summarization_chain(llm=None) -> Runnable:
    return _SUMMARY_PROMPT | (llm or get_llm()) | StrOutputParser()


retrieval_chain = build_retrieval_chain()
summarization_chain = build_summarization_chain()


@dataclass
class ConversationMemory:
    summary: str = ""
    recent_messages: list[BaseMessage] = field(default_factory=list)


def record_turn(
    memory: ConversationMemory, question: str, answer: str, summarizer: Runnable
) -> None:
    """Append the turn, then fold anything beyond the last MAX_RECENT_MESSAGES into the summary."""
    memory.recent_messages.append(HumanMessage(content=question))
    memory.recent_messages.append(AIMessage(content=answer))
    if len(memory.recent_messages) > MAX_RECENT_MESSAGES:
        overflow = memory.recent_messages[:-MAX_RECENT_MESSAGES]
        memory.recent_messages = memory.recent_messages[-MAX_RECENT_MESSAGES:]
        memory.summary = summarizer.invoke(
            {"existing_summary": memory.summary, "turns_to_summarize": overflow}
        )