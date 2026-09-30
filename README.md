# LineMate

A kitchen operations API with grounded Q&A and citations.

LineMate puts recipes, SOPs and incident reports behind one API. It flags what needs attention and answers kitchen questions from the documents themselves. It runs entirely on your machine: no cloud, no API keys, no database.

A Ticket here is an operational issue (broken equipment, supply shortage), not a customer order.

## Features

- REST API for documents, tickets, comments and crew (FastAPI, Pydantic v2)
- Stale documents: overdue for review (over 90 days; Incident Reports exempt)
- Mismatches: open tickets assigned outside the station that owns the document
- Workload analytics: open tickets by station and priority (pandas, numpy)
- Grounded Q&A: cited answers that warn when a source is out of date
- Memory: follow-up questions keep their context

## How it works

Documents are chunked, embedded and stored in Chroma. A question is rewritten if it is a follow-up, matched to chunks, and answered by a local LLM using only those sources. Citations are built from retrieval metadata, and the stale warning is added in code.

## Quick start

Requires Python 3.11+ and Ollama.

    pip install -r requirements.txt
    ollama pull llama3.2
    ollama pull nomic-embed-text

    python -m scripts.ingest
    uvicorn app.api.main:app --reload

Open http://127.0.0.1:8000/docs, click Authorize, and use the key `linemate-local-key`.

## Key endpoints

- `/documents`, `/documents/stale`: documents and overdue reviews
- `/tickets`, `/tickets/mismatches`: tickets and station mismatches
- `/analytics`: station workload
- `/ask`: question with citations
- `/ask/conversation`: question with memory

## Tests

    pytest -v

The LLM is faked in the tests, so Ollama is not needed.

## Limitations

Changes live in memory only, and there is one shared API key. New documents need `POST /ask/reindex`. The 3B model's query rewrites can vary between runs.