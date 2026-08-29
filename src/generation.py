import ollama

from config import LLM_MODEL


def generate_answer(question, retrieved_results):
    """
    Generate an answer using retrieved chunks and
    return the answer along with the source numbers
    cited by the model.
    """

    context_parts = []

    for index, result in enumerate(retrieved_results, start=1):
        context_parts.append(
            f"[SOURCE {index}]\n"
            f"File: {result['source']}\n"
            f"Content:\n{result['document']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an employee onboarding assistant.

Answer the user's question using ONLY the information
provided in the company documents below.

If the documents do not contain enough information to
answer the question, say exactly:

I couldn't find this information in the available company documents.

Do not use outside knowledge.

At the end of your answer, provide the source numbers
you actually used.

Use this exact format:

ANSWER:
<your answer>

SOURCES:
<source numbers separated by commas>

For example:

ANSWER:
Employees are entitled to 18 days of annual leave per year.

SOURCES:
1

Company documents:

{context}

User question:
{question}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output = response["message"]["content"].strip()

    return parse_generation(output, retrieved_results)


def parse_generation(output, retrieved_results):
    """
    Extract the answer and source numbers from
    the LLM response.
    """

    answer = output
    source_numbers = []

    if "ANSWER:" in output:
        answer = output.split("ANSWER:", 1)[1]

    if "SOURCES:" in answer:
        answer, sources_part = answer.split(
            "SOURCES:",
            1
        )

        for value in sources_part.split(","):
            value = value.strip()

            try:
                number = int(value)

                if 1 <= number <= len(retrieved_results):
                    source_numbers.append(number)

            except ValueError:
                continue

    answer = answer.strip()

    return {
        "answer": answer,
        "source_numbers": source_numbers
    }