"""Incrementally synchronize Version 3 chunks with Chroma."""

from pathlib import Path
import re
import time

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
        response = _embed_batch(batch, settings)
        vectors.extend(response["embeddings"])
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} new chunks")
    return vectors


def _embed_batch(batch: list[str], settings: Settings) -> dict:
    for attempt in range(settings.embedding_max_retries + 1):
        try:
            return embed(
                model=settings.embedding_model,
                input=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except Exception as error:
            delay = _quota_retry_delay(error, attempt, settings)
            if delay is None or attempt == settings.embedding_max_retries:
                raise
            print(
                f"Embedding quota reached; retrying this batch in {delay:.1f} seconds "
                f"({attempt + 1}/{settings.embedding_max_retries})"
            )
            time.sleep(delay)
    raise RuntimeError("Embedding retry loop ended unexpectedly")


def _quota_retry_delay(
    error: Exception, attempt: int, settings: Settings
) -> float | None:
    message = str(error)
    code = getattr(error, "code", None)
    if code != 429 and not re.search(r"429|RESOURCE_EXHAUSTED|quota", message, re.I):
        return None
    matches = re.findall(
        r"(?:retry in\s+|retryDelay['\"\s:]+)([\d.]+)s", message, re.I
    )
    suggested = max(map(float, matches), default=0.0)
    if suggested > settings.embedding_retry_max_seconds:
        return None
    exponential = settings.embedding_retry_base_seconds * (2**attempt)
    return min(
        max(suggested + 1.0, exponential),
        settings.embedding_retry_max_seconds,
    )


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
