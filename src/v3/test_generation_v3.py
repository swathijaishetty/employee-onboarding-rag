"""Offline checks for citation parsing and numeric grounding."""

from . import generation_v3
from .models_v3 import SearchResult


evidence = [SearchResult("id", "Enrollment is required within 30 days.", "benefits.pdf", 1, "Enrollment")]
original_chat = generation_v3.chat
try:
    captured = {}

    def grounded_chat(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return {"message": {"content": "Enroll within 30 days [1]."}}

    generation_v3.chat = grounded_chat
    result = generation_v3.generate("When do I enroll?", evidence)
    assert result == {"answer": "Enroll within 30 days [1].", "citations": [1]}
    assert "including table-like rows extracted from PDFs" in captured["prompt"]
    generation_v3.chat = lambda **_: {"message": {"content": "ANSWER:\nEnroll within 30 days.\nSOURCES:\n1"}}
    assert generation_v3.generate("When do I enroll?", evidence) == {
        "answer": "Enroll within 30 days. [1]", "citations": [1]
    }
    generation_v3.chat = lambda **_: {"message": {"content": "Enroll within 45 days [1]."}}
    assert generation_v3.generate("When do I enroll?", evidence)["citations"] == []
finally:
    generation_v3.chat = original_chat
extra = SearchResult("other", "Managers approve requests.", "other.pdf", 1, "Requests")
assert generation_v3._supported_citations(
    "Enrollment is required within 30 days.", [1, 2], [evidence[0], extra]
) == [1]

strong = SearchResult(
    "leave",
    "Full-time employees receive 18 days of annual leave per calendar year.",
    "leave.pdf",
    1,
    "Annual Leave",
    semantic_distance=0.2,
    lexical_score=12.0,
)
responses = iter(
    [
        {"message": {"content": generation_v3.FALLBACK}},
        {"message": {"content": "Full-time employees receive 18 days [1]."}},
    ]
)
retry_prompts = []
original_chat = generation_v3.chat
try:
    def retry_chat(**kwargs):
        retry_prompts.append(kwargs["messages"][0]["content"])
        return next(responses)

    generation_v3.chat = retry_chat
    recovered = generation_v3.generate(
        "How many annual leave days do full-time employees receive?", [strong]
    )
    assert recovered == {
        "answer": "Full-time employees receive 18 days [1].",
        "citations": [1],
    }
    assert len(retry_prompts) == 2
    assert "STRONGEST EVIDENCE" in retry_prompts[1]
finally:
    generation_v3.chat = original_chat
print("generation assertions: ok")
