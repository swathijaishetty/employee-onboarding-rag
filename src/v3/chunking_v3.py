"""Structure-aware, adaptive chunking with stable identifiers."""

from hashlib import sha256
import re

from .config_v3 import SETTINGS, Settings
from .models_v3 import Chunk, Page
from .pdf_reader_v3 import read_all_pdfs


NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]\s+\S")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
PAGE_HEADER = re.compile(r"^Fictional policy for RAG evaluation \| Page \d+$", re.I)


def _content_lines(page: Page) -> list[str]:
    lines = [line for line in page.text.splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        if line == "Northstar Technologies" or PAGE_HEADER.match(line):
            continue
        if page.policy_id and line.startswith(f"{page.policy_id} |"):
            continue
        cleaned.append(line)
    return cleaned


def _sections(page: Page) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Document overview"
    body: list[str] = []
    for line in _content_lines(page):
        if NUMBERED_HEADING.match(line):
            if body:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line, [line]
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body).strip()))
    sections = [(name, text) for name, text in sections if text]
    if len(sections) > 1 and sections[0][0] == "Document overview":
        preamble = sections[0][1]
        if len(preamble.split()) < 40:
            first_heading, first_text = sections[1]
            sections[1] = (first_heading, f"{preamble}\n{first_text}")
            sections.pop(0)
    return sections


def _adaptive_parts(text: str, settings: Settings) -> list[str]:
    """Keep a section whole unless it exceeds the configured safety ceiling."""
    if len(text.split()) <= settings.max_chunk_words:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    units = paragraphs if len(paragraphs) > 1 else [
        sentence.strip() for sentence in SENTENCE.split(text) if sentence.strip()
    ]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        unit_words = len(unit.split())
        if current and count + unit_words > settings.max_chunk_words:
            parts.append("\n".join(current))
            overlap = current[-settings.chunk_overlap_sentences :]
            current = overlap.copy()
            count = sum(len(item.split()) for item in current)
        current.append(unit)
        count += unit_words
    if current:
        parts.append("\n".join(current))
    return parts


def create_chunks(directory=None, settings: Settings = SETTINGS) -> list[Chunk]:
    pages = read_all_pdfs(directory or settings.pdf_directory)
    chunks: list[Chunk] = []
    for page in pages:
        for section_index, (section, text) in enumerate(_sections(page), start=1):
            parent_key = f"{page.source}|{page.page_number}|{section_index}|{section}"
            parent_id = sha256(parent_key.encode()).hexdigest()[:16]
            for part_index, part in enumerate(_adaptive_parts(text, settings), start=1):
                digest = sha256((parent_key + "|" + part).encode()).hexdigest()
                metadata: dict[str, str | int] = {
                    "source": page.source,
                    "page": page.page_number,
                    "section": section,
                    "section_index": section_index,
                    "part_index": part_index,
                    "parent_id": parent_id,
                    "document_hash": page.document_hash,
                    "policy_id": page.policy_id,
                    "title": page.title,
                    "word_count": len(part.split()),
                }
                chunks.append(Chunk(digest, part, metadata))
    return chunks
