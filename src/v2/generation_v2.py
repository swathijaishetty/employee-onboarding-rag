"""Grounded answer generation for the Version 2 assistant."""

import re

import ollama

try:
    from .config_v2 import LLM_MODEL
except ImportError:  # pragma: no cover
    from config_v2 import LLM_MODEL
FALLBACK_ANSWER = "I couldn't find that information in the available employee documents."


def _build_context(retrieved_documents: list[dict]) -> str:
    sections = []
    for index, result in enumerate(retrieved_documents, start=1):
        sections.append(
            f"[SOURCE {index}] {result['source']} (page {result['page']})\n"
            f"{result['document']}"
        )
    return "\n\n".join(sections)


def generate_answer(question: str, retrieved_documents: list[dict]) -> dict:
    """Generate a grounded answer and return only the sources cited by the model."""

    if not retrieved_documents:
        return {"answer": FALLBACK_ANSWER, "source_numbers": []}

    prompt = f"""You are the Northstar Technologies employee onboarding assistant.
Answer the employee's question using only the policy excerpts below.
Do not use outside knowledge or invent numbers, dates, eligibility rules, or procedures.
If the excerpts do not answer the question, use this exact sentence:
{FALLBACK_ANSWER}
Keep the answer concise and practical. Cite every excerpt that supports the answer.

Return exactly this format:
ANSWER:
<answer>

SOURCES:
<comma-separated source numbers, or none>

POLICY EXCERPTS:
{_build_context(retrieved_documents)}

EMPLOYEE QUESTION:
{question.strip()}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response["message"]["content"].strip()
    return parse_generation(output, len(retrieved_documents))


def parse_generation(output: str, source_count: int) -> dict:
    """Parse the required answer/source format defensively."""

    answer = output.strip()
    source_numbers: list[int] = []

    if "ANSWER:" in answer:
        answer = answer.split("ANSWER:", 1)[1]
    if "SOURCES:" in answer:
        answer, sources_part = answer.split("SOURCES:", 1)
        for value in re.findall(r"\d+", sources_part):
            number = int(value)
            if 1 <= number <= source_count and number not in source_numbers:
                source_numbers.append(number)

    answer = answer.strip()
    if not answer:
        answer = FALLBACK_ANSWER
    return {"answer": answer, "source_numbers": source_numbers}
