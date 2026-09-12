"""Shared data models for the Version 3 pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    source: str
    page_number: int
    text: str
    document_hash: str
    policy_id: str = ""
    title: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source: str
    page: int
    section: str
    policy_id: str = ""
    semantic_distance: float | None = None
    lexical_score: float = 0.0
    fused_score: float = 0.0
    metadata: dict = field(default_factory=dict)
