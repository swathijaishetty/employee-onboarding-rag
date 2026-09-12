"""Offline smoke checks for extraction and adaptive chunking."""

from .chunking_v3 import create_chunks


chunks = create_chunks()
assert chunks
assert all(chunk.text.strip() for chunk in chunks)
assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
assert all(chunk.metadata["source"].endswith(".pdf") for chunk in chunks)
assert all(int(chunk.metadata["page"]) >= 1 for chunk in chunks)
print(f"chunking assertions: ok ({len(chunks)} chunks, {len({c.metadata['source'] for c in chunks})} PDFs)")
