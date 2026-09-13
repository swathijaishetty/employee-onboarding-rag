"""Validated, Version 3-specific configuration."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _provider() -> str:
    value = os.getenv("V3_MODEL_PROVIDER", "ollama").strip().lower()
    if value not in {"ollama", "gemini"}:
        raise ValueError("V3_MODEL_PROVIDER must be 'ollama' or 'gemini'")
    return value


def _model(name: str, ollama_default: str, gemini_default: str) -> str:
    provider = os.getenv("V3_MODEL_PROVIDER", "ollama").strip().lower()
    return os.getenv(name, gemini_default if provider == "gemini" else ollama_default).strip()


@dataclass(frozen=True)
class Settings:
    pdf_directory: Path = Path(os.getenv("V3_PDF_DIRECTORY", "data/documents/PDF Files"))
    chroma_path: str = os.getenv("V3_CHROMA_PATH", "chroma_db_v3")
    collection_name: str = os.getenv("V3_COLLECTION_NAME", "employee_policies_v3")
    model_provider: str = _provider()
    llm_model: str = _model("V3_LLM_MODEL", "llama3.2:3b", "gemini-2.5-flash")
    embedding_model: str = _model(
        "V3_EMBEDDING_MODEL", "nomic-embed-text", "gemini-embedding-001"
    )
    ollama_host: str = os.getenv("V3_OLLAMA_HOST", "").strip()
    ollama_api_key: str = field(
        default=os.getenv("OLLAMA_API_KEY", "").strip(), repr=False
    )
    gemini_api_key: str = field(
        default=os.getenv("GEMINI_API_KEY", "").strip(), repr=False
    )
    embedding_batch_size: int = _integer("V3_EMBEDDING_BATCH_SIZE", 16)
    embedding_max_retries: int = _integer("V3_EMBEDDING_MAX_RETRIES", 6)
    embedding_retry_base_seconds: float = _number(
        "V3_EMBEDDING_RETRY_BASE_SECONDS", 2.0, 0.1, 60.0
    )
    embedding_retry_max_seconds: float = _number(
        "V3_EMBEDDING_RETRY_MAX_SECONDS", 60.0, 1.0, 600.0
    )
    max_chunk_words: int = _integer("V3_MAX_CHUNK_WORDS", 240, 80)
    chunk_overlap_sentences: int = _integer("V3_CHUNK_OVERLAP_SENTENCES", 1)
    semantic_candidates: int = _integer("V3_SEMANTIC_CANDIDATES", 24)
    lexical_candidates: int = _integer("V3_LEXICAL_CANDIDATES", 24)
    top_k: int = _integer("V3_TOP_K", 6)
    max_distance: float = _number("V3_MAX_DISTANCE", 0.85, 0.0, 2.0)
    semantic_only_max_distance: float = _number(
        "V3_SEMANTIC_ONLY_MAX_DISTANCE", 0.35, 0.0, 2.0
    )
    rrf_constant: int = _integer("V3_RRF_CONSTANT", 60)
    diversity_weight: float = _number("V3_DIVERSITY_WEIGHT", 0.18, 0.0, 1.0)
    max_chunks_per_source: int = _integer("V3_MAX_CHUNKS_PER_SOURCE", 2)
    memory_turns: int = _integer("V3_MEMORY_TURNS", 8)
    max_context_chars: int = _integer("V3_MAX_CONTEXT_CHARS", 12000, 1000)
    api_host: str = os.getenv("V3_API_HOST", "127.0.0.1")
    api_port: int = _integer("V3_API_PORT", 8000)
    max_question_chars: int = _integer("V3_MAX_QUESTION_CHARS", 2000, 100)
    session_ttl_seconds: int = _integer("V3_SESSION_TTL_SECONDS", 3600, 60)
    max_sessions: int = _integer("V3_MAX_SESSIONS", 500)
    requests_per_minute: int = _integer("V3_REQUESTS_PER_MINUTE", 12)


SETTINGS = Settings()
