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
            f"[SOURCE {index}] {result['source']} (page {result['page']}, "
            f"section: {result.get('section', 'Unknown')})\n"
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
Every factual claim must be directly supported by an excerpt. Copy limits, dates,
and numbers exactly as written; never infer an additional condition.
Do not convert a qualitative rule into a new number. For example, if a policy
says something does not carry forward, repeat that wording instead of saying zero.
When the question names a specific leave or benefit type, answer only for that
type unless a comparison is explicitly requested.
If the excerpts do not answer the question, use this exact sentence:
{FALLBACK_ANSWER}
Keep the answer concise and practical. Cite every excerpt that supports the answer.
If the excerpts answer the question, SOURCES must contain at least one source number.

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
        options={"temperature": 0},
    )
    output = response["message"]["content"].strip()
    result = parse_generation(output, len(retrieved_documents))

    # Small local models occasionally cite a filename despite the requested
    # numeric format. Resolve that citation against the retrieved evidence.
    if not result["source_numbers"]:
        for index, item in enumerate(retrieved_documents, start=1):
            if item["source"].lower() in output.lower():
                result["source_numbers"].append(index)

    # Reject numeric claims that do not occur anywhere in the retrieved evidence.
    evidence = " ".join(item["document"] for item in retrieved_documents)
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", result["answer"]))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
    if not answer_numbers.issubset(evidence_numbers):
        return {"answer": FALLBACK_ANSWER, "source_numbers": []}
    if result["answer"] != FALLBACK_ANSWER and not result["source_numbers"]:
        return {"answer": FALLBACK_ANSWER, "source_numbers": []}
    return result


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
