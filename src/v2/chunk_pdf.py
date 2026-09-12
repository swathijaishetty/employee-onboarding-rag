"""Create section-aware chunks from the onboarding policy PDFs."""

from pathlib import Path
import math
import re

from langchain_core.documents import Document

try:  # Support module and direct execution.
    from .config_v2 import MAX_SECTION_CHARS, MIN_SECTION_CHUNKS, PDF_DIRECTORY
    from .pdf_reader import read_all_pdfs
except ImportError:  # pragma: no cover
    from config_v2 import MAX_SECTION_CHARS, MIN_SECTION_CHUNKS, PDF_DIRECTORY
    from pdf_reader import read_all_pdfs


SECTION_HEADER = re.compile(r"^\d+\.\s+.+$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a page at numbered policy headings and retain the heading text."""

    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current: list[str] = []
    current_header = "Overview"

    for line in lines:
        if SECTION_HEADER.match(line.strip()):
            if current and "\n".join(current).strip():
                sections.append((current_header, "\n".join(current).strip()))
            current_header = line.strip()
            current = [line.strip()]
        else:
            current.append(line)

    if current and "\n".join(current).strip():
        sections.append((current_header, "\n".join(current).strip()))
    return sections


def _split_long_section(section: str) -> list[str]:
    """Split only oversized sections at sentence boundaries with light overlap."""

    if len(section) <= MAX_SECTION_CHARS:
        return [section]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", section)
        if sentence.strip()
    ]
    chunk_count = max(MIN_SECTION_CHUNKS, math.ceil(len(section) / MAX_SECTION_CHARS))
    target_chars = math.ceil(len(section) / chunk_count)

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        if current and current_length + len(sentence) + 1 > target_chars:
            chunks.append(" ".join(current).strip())
            # Carry the final sentence forward so a boundary keeps context.
            current = current[-1:]
            current_length = len(current[0])
        current.append(sentence)
        current_length += len(sentence) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def create_chunks(pdf_directory: str | Path = PDF_DIRECTORY) -> list[Document]:
    """Read pages and create chunks based on policy structure, not page length."""

    chunks: list[Document] = []
    for page in read_all_pdfs(pdf_directory):
        page_chunk_number = 0
        for section_number, (section_header, section_text) in enumerate(
            _split_sections(page.text), start=1
        ):
            for part_number, content in enumerate(
                _split_long_section(section_text), start=1
            ):
                page_chunk_number += 1
                chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": page.source,
                            "page": page.page_number,
                            "section": section_header,
                            "section_number": section_number,
                            "part_number": part_number,
                            "chunk_number": page_chunk_number,
                        },
                    )
                )

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return chunks


def inspect_chunks(chunks: list[Document]) -> None:
    """Print a compact preview useful when reviewing section boundaries."""

    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:12]:
        metadata = chunk.metadata
        print("\n" + "=" * 70)
        print(
            f"Chunk {metadata['chunk_number']} | {metadata['source']} | "
            f"page {metadata['page']} | {metadata['section']}"
        )
        print(chunk.page_content)


if __name__ == "__main__":
    inspect_chunks(create_chunks())
