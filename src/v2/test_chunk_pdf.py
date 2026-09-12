try:
    from .chunk_pdf import create_chunks
except ImportError:  # pragma: no cover
    from chunk_pdf import create_chunks


chunks = create_chunks()
print(f"\nCreated {len(chunks)} chunks from data/documents/PDF Files.")
for chunk in chunks[:10]:
    metadata = chunk.metadata
    print("\n" + "=" * 70)
    print(
        f"Source: {metadata['source']} | page {metadata['page']} | "
        f"chunk {metadata['chunk_number']}"
    )
    print(chunk.page_content[:500])
