"""Interactive CLI for Version 3."""

from .generation_v3 import generate
from .memory_v3 import SessionMemory
from .retrieval_v3 import retrieve


def answer_question(question: str, memory: SessionMemory) -> dict:
    search_query = memory.resolve(question)
    evidence = retrieve(search_query)
    result = generate(question, evidence)
    result["search_query"] = search_query
    result["evidence"] = evidence
    memory.add(question, search_query, result["answer"])
    return result


def main() -> None:
    print("=" * 68)
    print("Northstar Technologies - Employee Policy Assistant (Version 3)")
    print("=" * 68)
    print("Ask about the indexed PDF policies. Commands: clear, debug, exit.\n")
    memory = SessionMemory()
    debug = False
    while True:
        question = input("You: ").strip()
        command = question.lower()
        if command in {"exit", "quit"}:
            print("Goodbye!")
            return
        if command in {"clear", "reset"}:
            memory.clear()
            print("Assistant: Session context cleared.\n")
            continue
        if command == "debug":
            debug = not debug
            print(f"Assistant: Retrieval debug {'enabled' if debug else 'disabled'}.\n")
            continue
        if not question:
            continue
        try:
            result = answer_question(question, memory)
            print(f"\nAssistant: {result['answer']}")
            if debug:
                print(f"\nSearch query: {result['search_query']}")
                for rank, item in enumerate(result["evidence"], start=1):
                    distance = "n/a" if item.semantic_distance is None else f"{item.semantic_distance:.3f}"
                    print(f"  {rank}. {item.source} p.{item.page} | {item.section} | cosine distance={distance} | BM25={item.lexical_score:.2f}")
            citations = result["citations"]
            if citations:
                print("\nSources:")
                for number in citations:
                    item = result["evidence"][number - 1]
                    print(f"- [{number}] {item.source}, page {item.page}, {item.section}")
            print()
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
