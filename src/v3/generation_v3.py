"""Evidence-bound response generation with verified inline citations."""

import re

from .config_v3 import SETTINGS, Settings
from .model_client_v3 import chat
from .models_v3 import SearchResult
from .retrieval_v3 import bm25_scores


FALLBACK = "I couldn't find that information in the available employee documents."


def _supported_citations(answer: str, citations: list[int], results: list[SearchResult]) -> list[int]:
    """Discard model citations whose excerpts have weak lexical claim support."""
    if not citations:
        return []
    scores = bm25_scores(re.sub(r"\[\d+\]", "", answer), [item.text for item in results])
    strongest = max((scores[number - 1] for number in citations), default=0.0)
    if strongest <= 0:
        return []
    return [number for number in citations if scores[number - 1] >= strongest * 0.3]


def _context(results: list[SearchResult], limit: int) -> str:
    blocks: list[str] = []
    used = 0
    for number, item in enumerate(results, start=1):
        block = f"[{number}] {item.source}, page {item.page}, {item.section}\n{item.text}"
        if blocks and used + len(block) > limit:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def generate(question: str, results: list[SearchResult], settings: Settings = SETTINGS) -> dict:
    if not results:
        return {"answer": FALLBACK, "citations": []}
    prompt = f"""You are Northstar Technologies' employee policy assistant.
Answer only from the evidence. Explain conditions, exceptions, deadlines, and
employee categories that materially affect the answer. If evidence conflicts,
state the conflict instead of choosing silently. Do not infer missing facts.
Before returning the insufficient-evidence response, inspect every evidence
block, including table-like rows extracted from PDFs. Treat an explicit fact as
support even when the question uses different wording from the evidence.
For comparisons, state each supported fact side by side. Do not label one rule
stricter, better, earlier, or more generous unless the evidence says so.
Cite every factual sentence with one or more evidence numbers like [1]. If the
evidence is insufficient, return exactly: {FALLBACK}

Return exactly this structure:
ANSWER:
<answer with inline citations>
SOURCES:
<comma-separated evidence numbers, or none>

EVIDENCE:
{_context(results, settings.max_context_chars)}

QUESTION: {question}
"""
    response = chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    output = response["message"]["content"].strip()
    answer = output
    sources_part = ""
    if "ANSWER:" in answer:
        answer = answer.split("ANSWER:", 1)[1]
    if "SOURCES:" in answer:
        answer, sources_part = answer.split("SOURCES:", 1)
    answer = answer.strip()
    cited_values = re.findall(r"\[(\d+)\]", answer) + re.findall(r"\b(\d+)\b", sources_part)
    citations = sorted({int(value) for value in cited_values if 1 <= int(value) <= len(results)})
    citations = _supported_citations(answer, citations, results)
    if answer != FALLBACK and not citations:
        return {"answer": FALLBACK, "citations": []}
    allowed = set(citations)
    answer = re.sub(
        r"\s*\[(\d+)\]",
        lambda match: match.group(0) if int(match.group(1)) in allowed else "",
        answer,
    ).strip()
    if citations and not re.search(r"\[\d+\]", answer):
        answer += " " + " ".join(f"[{number}]" for number in citations)
    evidence = " ".join(item.text for item in results)
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", re.sub(r"\[\d+\]", "", answer)))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
    if not answer_numbers.issubset(evidence_numbers):
        return {"answer": FALLBACK, "citations": []}
    return {"answer": answer, "citations": citations}
