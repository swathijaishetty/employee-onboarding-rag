"""Incrementally synchronize Version 3 chunks with Chroma."""

from pathlib import Path

import chromadb

from .chunking_v3 import create_chunks
from .config_v3 import SETTINGS, Settings
from .model_client_v3 import embed


def get_collection(settings: Settings = SETTINGS):
    client = chromadb.PersistentClient(path=settings.chroma_path)
    expected_metadata = {
        "hnsw:space": "cosine",
        "pipeline_version": "3",
        "model_provider": settings.model_provider,
        "embedding_model": settings.embedding_model,
    }
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata=expected_metadata,
    )
    metadata = collection.metadata or {}
    for key in ("model_provider", "embedding_model"):
        if metadata.get(key) and metadata[key] != expected_metadata[key]:
            raise ValueError(
                f"Collection {settings.collection_name!r} was built with "
                f"{key}={metadata[key]!r}; configure a separate collection or rebuild it"
            )
    return collection


def _embed(texts: list[str], settings: Settings) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        response = embed(
            model=settings.embedding_model,
            input=batch,
            task_type="RETRIEVAL_DOCUMENT",
        )
        vectors.extend(response["embeddings"])
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} new chunks")
    return vectors


def ingest_documents(directory: str | Path | None = None, settings: Settings = SETTINGS) -> dict:
    chunks = create_chunks(directory, settings)
    if not chunks:
        raise ValueError("No extractable PDF content was found")
    collection = get_collection(settings)
    existing_ids = set(collection.get(include=[])["ids"])
    current_ids = {chunk.chunk_id for chunk in chunks}
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    stale_ids = sorted(existing_ids - current_ids)
    if new_chunks:
        embeddings = _embed([chunk.text for chunk in new_chunks], settings)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in new_chunks],
            documents=[chunk.text for chunk in new_chunks],
            metadatas=[chunk.metadata for chunk in new_chunks],
            embeddings=embeddings,
        )
    if stale_ids:
        collection.delete(ids=stale_ids)
    summary = {
        "documents": len({chunk.metadata["source"] for chunk in chunks}),
        "chunks": collection.count(),
        "embedded": len(new_chunks),
        "removed": len(stale_ids),
    }
    print("Ingestion complete: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


if __name__ == "__main__":
    ingest_documents()
