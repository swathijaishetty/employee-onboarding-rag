from retrieval_v2 import retrieve_context


print("\n========================================")
print("       V2 RAG RETRIEVAL TEST")
print("========================================")
print("Ask a question. Type 'exit' to quit.\n")


while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("\nExiting...")
        break

    if not question:
        continue

    results = retrieve_context(question)

    print("\nRetrieved Results:")
    print("-" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk_number']}")
        print(f"Distance: {result['distance']:.4f}")

        print("\nText:")
        print(result["document"][:500])

    print()