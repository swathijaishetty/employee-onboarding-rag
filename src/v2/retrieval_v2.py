"""Retrieve relevant chunks from the Version 2 Chroma collection."""

import re

import chromadb
import ollama

try:
    from .config_v2 import (
        CHROMA_PATH,
        COLLECTION_NAME,
        DISTANCE_THRESHOLD,
        EMBEDDING_MODEL,
        LEXICAL_FALLBACK_DISTANCE,
        MAX_CHUNKS_PER_SOURCE,
        RETRIEVAL_CANDIDATE_MULTIPLIER,
        TOP_K,
    )
except ImportError:  # pragma: no cover
    from config_v2 import (
        CHROMA_PATH,
        COLLECTION_NAME,
        DISTANCE_THRESHOLD,
        EMBEDDING_MODEL,
        LEXICAL_FALLBACK_DISTANCE,
        MAX_CHUNKS_PER_SOURCE,
        RETRIEVAL_CANDIDATE_MULTIPLIER,
        TOP_K,
    )

STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "from", "get",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "the", "to",
    "what", "when", "where", "which", "who", "will", "with", "you",
    "company", "employee", "employees", "policy", "policies", "guide",
    "days", "day", "working", "questions", "information", "documents",
    "available", "receive",
}
ALIASES = {
    "salary": {"pay", "payroll", "payslip"},
    "paycheck": {"pay", "payroll", "payslip"},
    "vacation": {"leave", "annual"},
    "remote": {"home", "work", "telework"},
    "remotely": {"home", "work", "telework"},
    "enroll": {"enrollment"},
    "sign": {"enrollment", "account"},
}


def _query_terms(query: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }
    for term in list(terms):
        terms.update(ALIASES.get(term, set()))
    return terms


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[dict]:
    """Return deduplicated, thresholded chunks ranked by cosine distance."""

    query = query.strip()
    if not query:
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    n_results = min(
        max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k), collection.count()
    )
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    query_terms = _query_terms(query)
    candidates: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for document, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        metadata = metadata or {}
        source = str(metadata.get("source", "Unknown"))
        page = int(metadata.get("page", 0))
        chunk_number = int(metadata.get("chunk_number", 0))
        section = str(metadata.get("section", ""))
        key = (source, page, chunk_number)
        document_terms = set(re.findall(r"[a-z0-9]+", str(document).lower()))
        lexical_hits = len(query_terms & document_terms)
        # A lexical anchor prevents semantically loose matches for out-of-scope
        # questions (for example, a stock-ticker question against policy text).
        if key in seen or distance > distance_threshold:
            continue
        if lexical_hits == 0 and distance > min(
            distance_threshold, LEXICAL_FALLBACK_DISTANCE
        ):
            continue
        seen.add(key)
        candidates.append(
            {
                "document": document,
                "source": source,
                "page": page,
                "section": section,
                "chunk_number": chunk_number,
                "distance": float(distance),
                "lexical_hits": lexical_hits,
            }
        )

    # Prefer excerpts sharing the strongest set of terms with the question. This
    # keeps a leave answer from being diluted by a merely related onboarding
    # excerpt that happens to mention one generic word.
    if query_terms and candidates:
        strongest_match = max(item["lexical_hits"] for item in candidates)
        candidates = [
            item for item in candidates if item["lexical_hits"] == strongest_match
        ]
    candidates.sort(key=lambda item: (-item["lexical_hits"], item["distance"]))
    selected: list[dict] = []
    per_source: dict[str, int] = {}
    for item in candidates:
        if per_source.get(item["source"], 0) >= MAX_CHUNKS_PER_SOURCE:
            continue
        selected.append(item)
        per_source[item["source"]] = per_source.get(item["source"], 0) + 1
        if len(selected) >= top_k:
            break
    return selected


# Backward-compatible name used by the original V2 smoke script.
retrieve_context = retrieve_documents


def print_results(results: list[dict]) -> None:
    print("\nRetrieved chunks:")
    print("=" * 70)
    for rank, result in enumerate(results, start=1):
        print(
            f"\nResult {rank} | {result['source']} | page {result['page']} | "
            f"chunk {result['chunk_number']} | distance {result['distance']:.4f}"
        )
        print(result["document"])


if __name__ == "__main__":
    print_results(retrieve_documents(input("Ask a question: ")))
