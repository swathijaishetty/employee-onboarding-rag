"""Extract normalized pages and document metadata from every configured PDF."""

from hashlib import sha256
from pathlib import Path
import re

from pypdf import PdfReader

from .models_v3 import Page


POLICY_ID_PATTERN = re.compile(r"\b(?:HR|IT|FIN|SEC)-[A-Z]+-\d{3}\b")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ").replace("\u00ad", "")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _metadata(first_page_text: str, fallback_title: str) -> tuple[str, str]:
    policy_match = POLICY_ID_PATTERN.search(first_page_text)
    policy_id = policy_match.group(0) if policy_match else ""
    lines = [line for line in first_page_text.splitlines() if line]
    title = fallback_title.replace("_", " ").title()
    if policy_id:
        for line in lines[:8]:
            if policy_id in line and "|" in line:
                title = line.split("|", 1)[1].strip()
                break
    return policy_id, title


def read_pdf(path: str | Path) -> list[Page]:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    raw = pdf_path.read_bytes()
    document_hash = sha256(raw).hexdigest()
    reader = PdfReader(str(pdf_path))
    extracted = [normalize_text(page.extract_text() or "") for page in reader.pages]
    first_text = next((text for text in extracted if text), "")
    policy_id, title = _metadata(first_text, pdf_path.stem)
    return [
        Page(pdf_path.name, number, text, document_hash, policy_id, title)
        for number, text in enumerate(extracted, start=1)
        if text
    ]


def read_all_pdfs(directory: str | Path) -> list[Page]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {root}")
    paths = sorted(root.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in: {root}")
    return [page for path in paths for page in read_pdf(path)]
