"""PDF text extraction helpers for the Version 2 pipeline."""

from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFPage:
    text: str
    page_number: int
    source: str


def normalize_text(text: str) -> str:
    """Clean extracted PDF text while preserving useful line structure."""

    if not text:
        return ""

    text = (
        text.replace("\u00a0", " ")
        .replace("\u00ad", "")
        .replace("\ufffd", "")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin words split at a line-ending hyphen, but retain normal hyphens.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(pdf_path: str | Path) -> list[PDFPage]:
    """
    Read a PDF and return one PDFPage object per page.
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    pages: list[PDFPage] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        text = normalize_text(text)
        if text:
            pages.append(PDFPage(text, page_number, path.name))
    return pages


def read_all_pdfs(pdf_directory: str | Path) -> list[PDFPage]:
    """
    Read all PDFs from the specified directory.

    Only PDFs directly inside the supplied directory are read.
    """

    directory = Path(pdf_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {directory}")
    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")
    pages: list[PDFPage] = []
    for pdf_path in pdf_files:
        pages.extend(read_pdf(pdf_path))
    return pages


if __name__ == "__main__":
    directory = Path("data/documents/PDF Files")
    pages = read_all_pdfs(directory)
    print(f"PDF files: {len(sorted(directory.glob('*.pdf')))}")
    print(f"Pages with text: {len(pages)}")
    for page in pages:
        print(f"{page.source} | page {page.page_number} | {len(page.text)} characters")
