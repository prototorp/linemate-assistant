"""Central settings."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DOCS_DIR = BASE_DIR / "data" / "docs"
CREW_CSV = BASE_DIR / "data" / "crew.csv"
TICKETS_CSV = BASE_DIR / "data" / "tickets.csv"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION = "linemate_docs"

# Business rules
STALE_THRESHOLD_DAYS = 90
OPEN_STATUSES = ("Open", "In-Progress")
PRIORITY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_K = 4
MAX_RECENT_MESSAGES = 6

# Demo only API key
API_KEY = os.getenv("LINEMATE_API_KEY", "linemate-local-key")