"""Central configuration for the Version 2 PDF RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


PDF_DIRECTORY = Path(
    os.getenv("PDF_DIRECTORY", "data/documents/PDF Files")
)
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db_v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "employee_policies_v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

TOP_K = _int("TOP_K", "3")
DISTANCE_THRESHOLD = _float("DISTANCE_THRESHOLD", "0.85")
MEMORY_TURNS = _int("MEMORY_TURNS", "6")

# Chunking is section-first. This cap only protects against unusually large
# sections and is configurable for a different document style.
MAX_SECTION_CHARS = _int("MAX_SECTION_CHARS", "1200")
MIN_SECTION_CHUNKS = _int("MIN_SECTION_CHUNKS", "2")

RETRIEVAL_CANDIDATE_MULTIPLIER = _int(
    "RETRIEVAL_CANDIDATE_MULTIPLIER", "3"
)
MAX_CHUNKS_PER_SOURCE = _int("MAX_CHUNKS_PER_SOURCE", "2")
LEXICAL_FALLBACK_DISTANCE = _float("LEXICAL_FALLBACK_DISTANCE", "0.38")
MAX_REWRITE_CHARS = _int("MAX_REWRITE_CHARS", "500")
