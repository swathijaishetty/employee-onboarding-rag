import chromadb
import ollama


# -----------------------------
# 1. Connect to Chroma
# -----------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="employee_policies"
)


# -----------------------------
# 2. Start the chat
# -----------------------------

print("\n========================================")
print("      EMPLOYEE ONBOARDING ASSISTANT")
print("========================================")
print("Ask a question. Type 'exit' to quit.\n")


while True:

    # -----------------------------
    # 3. Get question from employee
    # -----------------------------

    question = input("You: ").strip().lower()

    if question == "exit":
        print("\nGoodbye! 👋")
        break

    greetings = ["hi", "hello", "hey", "hii", "helo", "hey there"]

    if question in greetings:
        print("\nAssistant: Hi! 👋 How can I help you with your onboarding?\n")
        continue


    # -----------------------------
    # 4. Create question embedding
    # -----------------------------

    response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    question_embedding = response["embeddings"][0]


    # -----------------------------
    # 5. Retrieve relevant chunks
    # -----------------------------

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]


    # -----------------------------
    # 6. Build context
    # -----------------------------

    context = "\n\n".join(retrieved_chunks)


    # -----------------------------
    # 7. Create prompt
    # -----------------------------

    prompt = f"""
You are an Employee Onboarding Assistant.

Answer the employee's question using ONLY the
information provided in the context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the available company documents."

Context:
{context}

Employee Question:
{question}

Answer:
"""


    # -----------------------------
    # 8. Generate answer
    # -----------------------------

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]


    # -----------------------------
    # 9. Display answer
    # -----------------------------

    print(f"\nAssistant: {answer}\n")