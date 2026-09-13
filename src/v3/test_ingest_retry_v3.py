"""Offline checks for quota-aware embedding retries."""

from .config_v3 import Settings
from . import ingest_v3


settings = Settings(
    embedding_batch_size=2,
    embedding_max_retries=2,
    embedding_retry_base_seconds=1,
    embedding_retry_max_seconds=30,
)
calls = 0
sleeps: list[float] = []
original_embed = ingest_v3.embed
original_sleep = ingest_v3.time.sleep


def fake_embed(**kwargs):
    global calls
    calls += 1
    if calls == 1:
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 14.5s.")
    return {"embeddings": [[0.1], [0.2]]}


try:
    ingest_v3.embed = fake_embed
    ingest_v3.time.sleep = sleeps.append
    assert ingest_v3._embed(["one", "two"], settings) == [[0.1], [0.2]]
    assert sleeps == [15.5]
    assert calls == 2
finally:
    ingest_v3.embed = original_embed
    ingest_v3.time.sleep = original_sleep

print("ingestion retry assertions: ok")
