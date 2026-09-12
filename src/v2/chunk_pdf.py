"""Create retrieval-friendly chunks from the six onboarding PDFs."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # Support module and direct execution.
    from .pdf_reader import read_all_pdfs
except ImportError:  # pragma: no cover
    from pdf_reader import read_all_pdfs


PDF_DIRECTORY = Path("data/documents/PDF Files")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def create_chunks(pdf_directory: str | Path = PDF_DIRECTORY) -> list[Document]:
    """Read policy pages and split them while retaining source/page metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        strip_whitespace=True,
    )

    chunks: list[Document] = []
    for page in read_all_pdfs(pdf_directory):
        page_document = Document(
            page_content=page.text,
            metadata={"source": page.source, "page": page.page_number},
        )
        for chunk_number, chunk in enumerate(
            splitter.split_documents([page_document]), start=1
        ):
            chunk.metadata["chunk_number"] = chunk_number
            chunks.append(chunk)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


def inspect_chunks(chunks: list[Document]) -> None:
    """Print a compact preview useful when adjusting chunk boundaries."""

    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:10]:
        metadata = chunk.metadata
        print("\n" + "=" * 70)
        print(
            f"Chunk {metadata['chunk_id']} | {metadata['source']} | "
            f"page {metadata['page']} | part {metadata['chunk_number']}"
        )
        print(chunk.page_content)


if __name__ == "__main__":
    chunks = create_chunks()
    inspect_chunks(chunks)
