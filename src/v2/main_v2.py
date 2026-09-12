"""Interactive CLI for the PDF-backed Version 2 assistant."""

try:  # Support module and direct execution.
    from .generation_v2 import generate_answer
    from .memory_v2 import SessionMemory
    from .retrieval_v2 import retrieve_documents
except ImportError:  # pragma: no cover
    from generation_v2 import generate_answer
    from memory_v2 import SessionMemory
    from retrieval_v2 import retrieve_documents


GREETINGS = {"hi", "hello", "hey", "hii", "helo", "hey there"}
ACKNOWLEDGEMENTS = {"thanks", "thank you", "thx", "ok", "okay", "got it"}


def main() -> None:
    print("=" * 60)
    print("Northstar Technologies - Employee Onboarding RAG")
    print("=" * 60)
    print("Ask about the six onboarding policy PDFs. Type 'exit' to quit.")
    print("Type 'clear' to reset follow-up context.\n")

    memory = SessionMemory()

    while True:
        question = input("You: ").strip()
        normalized = question.lower()

        if normalized in {"exit", "quit"}:
            print("Goodbye!")
            return
        if normalized in {"clear", "reset", "new session"}:
            memory.clear()
            print("\nAssistant: Follow-up context cleared.\n")
            continue
        if not question:
            continue
        if normalized in GREETINGS:
            print("\nAssistant: Hi! How can I help with your onboarding?\n")
            continue
        if normalized in ACKNOWLEDGEMENTS:
            print("\nAssistant: You're welcome!\n")
            continue

        try:
            search_question = memory.standalone_question(question)
            retrieved = retrieve_documents(search_question)
            result = generate_answer(search_question, retrieved)
            print(f"\nAssistant: {result['answer']}")

            cited = result["source_numbers"]
            if cited:
                print("\nSources:")
                seen_sources = set()
                for number in cited:
                    source = retrieved[number - 1]
                    source_key = (source["source"], source["page"])
                    if source_key not in seen_sources:
                        print(f"- {source['source']}, page {source['page']}")
                        seen_sources.add(source_key)
            memory.add_turn(question, result["answer"])
            print()
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
