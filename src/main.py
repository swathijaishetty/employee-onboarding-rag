from generation import generate_answer
from retrieval import retrieve_context


print("\n========================================")
print("      EMPLOYEE ONBOARDING ASSISTANT")
print("========================================")
print("Ask a question. Type 'exit' to quit.\n")


greetings = [
    "hi",
    "hello",
    "hey",
    "hii",
    "helo",
    "hey there"
]


acknowledgements = [
    "thanks",
    "thank you",
    "appreciate it",
    "thx",
    "ty",
    "okay",
    "ok",
    "alright",
    "got it",
    "great",
    "ok thanks"
]


while True:

    question = input("You: ").strip()

    # Exit
    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    # Ignore empty input
    if not question:
        continue

    # Greetings
    if question.lower() in greetings:
        print(
            "\nAssistant: Hi! 👋 "
            "How can I help you with your onboarding?\n"
        )
        continue

    # Acknowledgements
    if question.lower() in acknowledgements:
        print(
            "\nAssistant: You're welcome! 😊 "
            "Is there anything else I can help you with?\n"
        )
        continue

    # Retrieve relevant documents
    retrieved_results = retrieve_context(question)

    # Nothing relevant found
    if not retrieved_results:
        print(
            "\nAssistant: I couldn't find this information "
            "in the available company documents.\n"
        )
        continue

    # Generate answer
    result = generate_answer(
        question,
        retrieved_results
    )

    answer = result["answer"]
    source_numbers = result["source_numbers"]

    print(f"\nAssistant: {answer}")

    # Display ONLY sources cited by the LLM
    if source_numbers:

        print("\nSources:")

        displayed_sources = set()

        for number in source_numbers:

            source = retrieved_results[number - 1]["source"]

            if source not in displayed_sources:
                print(f"- {source}")
                displayed_sources.add(source)

    print()