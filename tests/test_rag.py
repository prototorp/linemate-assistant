"""RAG layer tests. No Ollama, no Chroma: the LLM and retriever are replaced with fakes, so what is
tested is OUR wiring: the LCEL chain, memory, citations and staleness flags."""

from datetime import date

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.rag import qa_chain
from app.rag.ask_service import AskService, build_citations, stale_note
from app.rag.qa_chain import NO_ANSWER, ConversationMemory, build_retrieval_chain, record_turn
from tests.fakes import FakeRetriever, make_chunk

AS_OF = date(2026, 9, 28)


class ScriptedLLM:
    """A fake chat model as a Runnable. Records every prompt it receives and answers by role:
    the condense prompt gets a rewritten question, everything else gets a cited answer."""

    def __init__(self, condensed="How often do we filter the fryers?", answer="Twice a day [1]."):
        self.prompts: list[str] = []
        self.condensed, self.answer = condensed, answer

    def as_runnable(self) -> Runnable:
        def respond(prompt_value):
            text = prompt_value.to_string()
            self.prompts.append(text)
            return AIMessage(content=self.condensed if "rewrite follow-up questions" in text.lower() else self.answer)
        return RunnableLambda(respond)

    @property
    def condense_calls(self):
        return [p for p in self.prompts if "rewrite follow-up questions" in p.lower()]

    @property
    def answer_calls(self):
        return [p for p in self.prompts if "numbered sources" in p.lower()]


def _chain(chunks, llm=None):
    llm = llm or ScriptedLLM()
    retriever = FakeRetriever(chunks)
    return build_retrieval_chain(llm.as_runnable(), retriever.factory), llm, retriever


# ---- the LCEL chain ------------------------------------------------------------
def test_chain_is_a_composed_runnable():
    chain, _, _ = _chain([make_chunk()])
    assert isinstance(chain, Runnable)
    assert isinstance(qa_chain.retrieval_chain, Runnable)  # the real module-level chain too


def test_first_question_skips_the_rewrite_step():
    chain, llm, retriever = _chain([make_chunk()])
    result = chain.invoke({"question": "How often do we filter the fryers?", "summary": "", "recent_turns": []})
    assert llm.condense_calls == []                                   # no history, no rewrite call
    assert retriever.queries == ["How often do we filter the fryers?"]
    assert result["standalone_question"] == "How often do we filter the fryers?"
    assert result["answer"] == "Twice a day [1]."
    assert [d.metadata["document_id"] for d in result["documents"]] == [2]


def test_chain_output_keeps_every_intermediate_key():
    chain, _, _ = _chain([make_chunk()])
    result = chain.invoke({"question": "fryer oil?", "summary": "", "recent_turns": []})
    assert {"question", "standalone_question", "documents", "context", "answer"} <= set(result)


def test_follow_up_is_rewritten_before_retrieval():
    from langchain_core.messages import HumanMessage
    chain, llm, retriever = _chain([make_chunk()], ScriptedLLM(condensed='"How often do we filter the fryers?"'))
    history = [HumanMessage("Tell me about fryer maintenance"), AIMessage("Filter and change the oil [1].")]
    result = chain.invoke({"question": "how often do we do that?", "summary": "", "recent_turns": history})
    assert len(llm.condense_calls) == 1
    assert retriever.queries == ["How often do we filter the fryers?"]   # rewritten and de-quoted, NOT "that"
    assert result["standalone_question"] == "How often do we filter the fryers?"


def test_bad_rewrite_falls_back_to_previous_question_plus_follow_up():
    from langchain_core.messages import HumanMessage
    # the model "answers" instead of rewriting, which is not a question
    chain, llm, retriever = _chain([make_chunk()], ScriptedLLM(condensed="36F to 38F"))
    history = [HumanMessage("What temperature should the walk-in be?"), AIMessage("36F to 38F [1].")]
    result = chain.invoke({"question": "what if it goes above that?", "summary": "", "recent_turns": history})
    expected = "What temperature should the walk-in be? what if it goes above that?"
    assert result["standalone_question"] == expected
    assert retriever.queries == [expected]


def test_answer_prompt_receives_history_context_and_stale_flag():
    from langchain_core.messages import HumanMessage
    chain, llm, _ = _chain([make_chunk(last_reviewed_at="2026-02-10")])
    chain.invoke({"question": "how often?", "summary": "Crew asked about fryers.",
                  "recent_turns": [HumanMessage("Tell me about fryers"), AIMessage("Okay.")]})
    prompt = llm.answer_calls[0]
    assert "Crew asked about fryers." in prompt      # summary reaches the model
    assert "Tell me about fryers" in prompt          # recent turns reach the model
    assert "[1] Fryer Oil Filtration and Change SOP" in prompt
    assert "STALE: not reviewed in" in prompt        # computed in code, written into the source header


def test_nothing_retrieved_returns_fixed_reply_without_calling_the_llm():
    chain, llm, _ = _chain([])
    result = chain.invoke({"question": "What is the wifi password?", "summary": "", "recent_turns": []})
    assert result["answer"] == NO_ANSWER
    assert llm.answer_calls == []                    # nothing to hallucinate from


def test_answer_prompt_keeps_its_template_variables():
    # guards against a stray { } in the prompt text breaking the template
    assert {"context", "summary", "question"} <= set(qa_chain._ANSWER_PROMPT.input_variables)


# ---- memory --------------------------------------------------------------------
def test_memory_keeps_recent_turns_without_summarising_early():
    memory, summarizer = ConversationMemory(), RunnableLambda(lambda x: "SUMMARY")
    record_turn(memory, "q1", "a1", summarizer)
    record_turn(memory, "q2", "a2", summarizer)
    assert len(memory.recent_messages) == 4 and memory.summary == ""


def test_overflow_is_folded_into_the_summary_oldest_first():
    seen = {}
    def summarize(x):
        seen.update(x)
        return "Crew asked about the fryers."
    memory, summarizer = ConversationMemory(), RunnableLambda(summarize)
    for i in range(1, 5):                                            # 4 turns = 8 messages, max is 6
        record_turn(memory, f"q{i}", f"a{i}", summarizer)
    assert len(memory.recent_messages) == qa_chain.MAX_RECENT_MESSAGES
    assert memory.recent_messages[0].content == "q2"                 # q1/a1 were folded away
    assert [m.content for m in seen["turns_to_summarize"]] == ["q1", "a1"]
    assert memory.summary == "Crew asked about the fryers."


# ---- citations -------------------------------------------------------------------
def test_citations_come_from_metadata_and_mark_which_were_used():
    chunks = [make_chunk(document_id=2), make_chunk(document_id=1, title="Steak Guide", last_reviewed_at="2026-09-02")]
    citations = build_citations(chunks, "Filter twice a day [1].", as_of=AS_OF)
    assert [c.title for c in citations] == ["Fryer Oil Filtration and Change SOP", "Steak Guide"]
    assert [c.cited_in_answer for c in citations] == [True, False]
    assert [c.is_stale for c in citations] == [True, False]


def test_two_chunks_of_one_document_produce_one_citation():
    chunks = [make_chunk(document_id=2, chunk_index=0, text="first part"),
              make_chunk(document_id=2, chunk_index=1, text="second part")]
    citations = build_citations(chunks, "It says so [2].", as_of=AS_OF)
    assert len(citations) == 1
    assert citations[0].cited_in_answer is True
    assert citations[0].source_numbers == [1, 2]         # both [n] labels map to this one document
    assert citations[0].excerpt == "second part"          # shows the chunk the answer used


def test_every_number_in_the_answer_maps_to_exactly_one_listed_document():
    chunks = [make_chunk(document_id=2, chunk_index=0), make_chunk(document_id=2, chunk_index=1),
              make_chunk(document_id=1, title="Steak Guide", last_reviewed_at="2026-09-02")]
    citations = build_citations(chunks, "Target is 36F [1] and [2]. Steaks rest 5 minutes [3].", as_of=AS_OF)
    assert [(c.document_id, c.source_numbers, c.cited_in_answer) for c in citations] == [
        (2, [1, 2], True),
        (1, [3], True),
    ]
    listed = [n for c in citations for n in c.source_numbers]
    assert sorted(listed) == [1, 2, 3]                   # no number is missing or repeated


def test_answer_that_cites_nothing_is_visible_in_the_citations():
    citations = build_citations([make_chunk()], "I made this up.", as_of=AS_OF)
    assert citations[0].cited_in_answer is False


def test_long_excerpts_are_truncated():
    citations = build_citations([make_chunk(text="word " * 200)], "x [1]", as_of=AS_OF)
    assert len(citations[0].excerpt) <= 243 and citations[0].excerpt.endswith("...")


# ---- stale note (added in code) ---------------------------------------------------
def test_stale_note_only_for_cited_stale_sources():
    chunks = [make_chunk(document_id=2, last_reviewed_at="2026-02-10"),
              make_chunk(document_id=1, title="Steak Guide", last_reviewed_at="2026-09-02"),
              make_chunk(document_id=3, title="Old Unused Doc", last_reviewed_at="2025-01-01")]
    citations = build_citations(chunks, "Filter twice a day [1]. Steaks rest 5 minutes [2].", as_of=AS_OF)
    note = stale_note(citations, as_of=AS_OF)
    assert "Fryer Oil Filtration and Change SOP" in note and "230 days" in note
    assert "Steak Guide" not in note          # cited but fresh
    assert "Old Unused Doc" not in note       # stale but never cited


def test_no_stale_note_when_nothing_stale_is_cited():
    citations = build_citations([make_chunk(last_reviewed_at="2026-09-02")], "Fine [1].", as_of=AS_OF)
    assert stale_note(citations, as_of=AS_OF) == ""
    incident = build_citations([make_chunk(category="Incident Report", last_reviewed_at="2020-01-01")],
                               "It happened [1].", as_of=AS_OF)
    assert stale_note(incident, as_of=AS_OF) == ""


# ---- AskService end to end (fake chain, real memory handling) -------------------------
def _service(llm=None, chunks=None):
    chain, llm, retriever = _chain(chunks or [make_chunk()], llm)
    return AskService(chain=chain, summarizer=RunnableLambda(lambda x: "SUMMARY")), llm, retriever


def test_service_follow_up_uses_the_previous_turn():
    service, llm, retriever = _service()
    first = service.ask_in_conversation("c1", "How often do we filter the fryers?")
    second = service.ask_in_conversation("c1", "and who owns that?")
    assert len(llm.condense_calls) == 1  # only turn 2 rewrites
    assert first.sources[0].title == "Fryer Oil Filtration and Change SOP"
    assert "How often do we filter the fryers?" in llm.condense_calls[0]   # turn 1 was in the history


def test_service_conversations_do_not_leak():
    service, llm, _ = _service()
    service.ask_in_conversation("a", "How often do we filter the fryers?")
    service.ask_in_conversation("b", "What is the steak rest time?")
    assert llm.condense_calls == []                  # b never saw a's history


def test_one_off_ask_has_no_memory():
    service, llm, _ = _service()
    service.ask("first question please")
    service.ask("second question please")
    assert llm.condense_calls == []


def test_clear_conversation_forgets_history():
    service, llm, _ = _service()
    service.ask_in_conversation("c1", "How often do we filter the fryers?")
    assert service.clear_conversation("c1") is True
    service.ask_in_conversation("c1", "fresh start question")
    assert llm.condense_calls == []
    assert service.clear_conversation("nope") is False


def test_service_appends_stale_note_but_memory_keeps_the_clean_answer():
    service, _, _ = _service(chunks=[make_chunk(last_reviewed_at="2026-02-10")])  # stale SOP
    result = service.ask_in_conversation("c1", "How often do we filter the fryers?")
    assert result.answer.startswith("Twice a day [1].")
    assert "Note:" in result.answer and "confirm with your sous chef" in result.answer
    assert service._conversations["c1"].recent_messages[-1].content == "Twice a day [1]."