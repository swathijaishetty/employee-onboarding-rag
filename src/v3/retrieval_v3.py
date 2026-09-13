"""Hybrid semantic/BM25 retrieval with reciprocal-rank fusion and diversity."""

from collections import Counter, defaultdict
import math
import re

from .config_v3 import SETTINGS, Settings
from .ingest_v3 import get_collection
from .model_client_v3 import embed
from .models_v3 import SearchResult


STOP_WORDS = frozenset(
    "a an and are as at be by can company do does document employee employees for from had has have how i in information is it me my of on or our policy policies the to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token[:-1] if token.endswith("s") and len(token) > 4 else token
            for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    if not tokenized:
        return []
    query_terms = set(tokenize(query))
    average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    frequencies = Counter(term for term in query_terms for doc in tokenized if term in set(doc))
    scores: list[float] = []
    for doc in tokenized:
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            document_frequency = frequencies[term]
            inverse_frequency = math.log(1 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5))
            frequency = counts[term]
            denominator = frequency + k1 * (1 - b + b * len(doc) / average_length)
            if denominator:
                score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _jaccard(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def _comparison_parts(query: str) -> list[str]:
    """Expose both sides of an explicit comparison for coverage selection."""
    if not re.search(r"\b(compare|comparison|versus|vs\.?)\b", query, re.I):
        return []
    cleaned = re.sub(r"^\s*compare\s+", "", query, flags=re.I)
    parts = re.split(r"\s+(?:and|versus|vs\.?)\s+", cleaned, maxsplit=1, flags=re.I)
    return [part.strip(" ?." ) for part in parts if len(tokenize(part)) >= 2] if len(parts) == 2 else []


def retrieve(query: str, settings: Settings = SETTINGS) -> list[SearchResult]:
    query = query.strip()
    if not query:
        return []
    collection = get_collection(settings)
    count = collection.count()
    if not count:
        return []
    catalog = collection.get(include=["documents", "metadatas"])
    ids = catalog["ids"]
    documents = catalog["documents"] or []
    metadatas = catalog["metadatas"] or []
    query_vector = embed(
        model=settings.embedding_model,
        input=query,
        task_type="RETRIEVAL_QUERY",
    )["embeddings"][0]
    semantic = collection.query(
        query_embeddings=[query_vector],
        n_results=min(settings.semantic_candidates, count),
        include=["distances"],
    )
    semantic_ids = semantic["ids"][0]
    distances = dict(zip(semantic_ids, semantic["distances"][0]))
    search_documents = [
        f"{metadata.get('title', '')} {metadata.get('section', '')} {document}"
        for document, metadata in zip(documents, metadatas)
    ]
    lexical_values = bm25_scores(query, search_documents)
    lexical_order = sorted(range(len(ids)), key=lambda i: lexical_values[i], reverse=True)
    lexical_ids = [ids[i] for i in lexical_order[: settings.lexical_candidates] if lexical_values[i] > 0]
    lexical_by_id = dict(zip(ids, lexical_values))
    if not lexical_ids and min(distances.values(), default=2.0) > settings.semantic_only_max_distance:
        return []
    fused: defaultdict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(semantic_ids, start=1):
        if distances[chunk_id] <= settings.max_distance:
            fused[chunk_id] += 1 / (settings.rrf_constant + rank)
    for rank, chunk_id in enumerate(lexical_ids, start=1):
        fused[chunk_id] += 1 / (settings.rrf_constant + rank)
    index = {chunk_id: i for i, chunk_id in enumerate(ids)}
    candidates: list[SearchResult] = []
    for chunk_id, score in fused.items():
        i = index[chunk_id]
        metadata = metadatas[i] or {}
        candidates.append(SearchResult(
            chunk_id=chunk_id,
            text=documents[i],
            source=str(metadata.get("source", "Unknown")),
            page=int(metadata.get("page", 0)),
            section=str(metadata.get("section", "Unknown")),
            policy_id=str(metadata.get("policy_id", "")),
            semantic_distance=distances.get(chunk_id),
            lexical_score=lexical_by_id.get(chunk_id, 0.0),
            fused_score=score,
            metadata=metadata,
        ))
    selected: list[SearchResult] = []
    per_source: Counter = Counter()
    candidate_by_id = {item.chunk_id: item for item in candidates}
    for part in _comparison_parts(query):
        part_scores = bm25_scores(part, search_documents)
        for i in sorted(range(len(ids)), key=lambda index: part_scores[index], reverse=True):
            item = candidate_by_id.get(ids[i])
            if not item or part_scores[i] <= 0 or per_source[item.source]:
                continue
            selected.append(item)
            per_source[item.source] += 1
            candidates.remove(item)
            break
    while candidates and len(selected) < settings.top_k:
        best = max(candidates, key=lambda item: item.fused_score - settings.diversity_weight * max(
            (_jaccard(item.text, prior.text) for prior in selected), default=0.0
        ))
        candidates.remove(best)
        if per_source[best.source] >= settings.max_chunks_per_source:
            continue
        selected.append(best)
        per_source[best.source] += 1
    return selected
