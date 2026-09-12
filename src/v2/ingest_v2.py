"""Ingest the six policy PDFs into a local Chroma collection."""

import os
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv

try:  # Support ``python -m src.v2.ingest_v2`` and direct execution.
    from .chunk_pdf import PDF_DIRECTORY, create_chunks
except ImportError:  # pragma: no cover
    from chunk_pdf import PDF_DIRECTORY, create_chunks


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db_v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "employee_policies_v2")


def get_collection():
    """Open the configured local collection, creating it with cosine distance."""

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for index, text in enumerate(texts, start=1):
        print(f"Creating embedding {index}/{len(texts)}")
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embeddings.append(response["embeddings"][0])
    return embeddings


def ingest_documents(pdf_directory: str | Path = PDF_DIRECTORY) -> int:
    """Synchronize PDF chunks and embeddings with the configured collection."""

    pdf_directory = Path(pdf_directory)
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if len(pdf_files) != 6:
        raise ValueError(
            f"Expected 6 policy PDFs in {pdf_directory}, found {len(pdf_files)}"
        )

    print(f"Reading {len(pdf_files)} PDFs from {pdf_directory}")
    chunks = create_chunks(pdf_directory)
    if not chunks:
        raise ValueError("No text chunks were extracted from the policy PDFs")
    print(f"Created {len(chunks)} chunks")

    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [
        f"{Path(metadata['source']).stem}-p{metadata['page']}-c{metadata['chunk_number']}"
        for metadata in metadatas
    ]

    collection = get_collection()
    embeddings = _embed_documents(documents)

    # Upsert current chunks and remove chunks from PDFs that were deleted/changed.
    existing_ids = set(collection.get(include=[])["ids"])
    stale_ids = existing_ids - set(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("\nIngestion completed successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks stored: {collection.count()}")
    return len(chunks)


if __name__ == "__main__":
    ingest_documents()
