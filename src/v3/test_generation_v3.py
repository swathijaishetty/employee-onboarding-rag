"""Offline checks for citation parsing and numeric grounding."""

from . import generation_v3
from .models_v3 import SearchResult


evidence = [SearchResult("id", "Enrollment is required within 30 days.", "benefits.pdf", 1, "Enrollment")]
original_chat = generation_v3.ollama.chat
try:
    generation_v3.ollama.chat = lambda **_: {"message": {"content": "Enroll within 30 days [1]."}}
    result = generation_v3.generate("When do I enroll?", evidence)
    assert result == {"answer": "Enroll within 30 days [1].", "citations": [1]}
    generation_v3.ollama.chat = lambda **_: {"message": {"content": "ANSWER:\nEnroll within 30 days.\nSOURCES:\n1"}}
    assert generation_v3.generate("When do I enroll?", evidence) == {
        "answer": "Enroll within 30 days. [1]", "citations": [1]
    }
    generation_v3.ollama.chat = lambda **_: {"message": {"content": "Enroll within 45 days [1]."}}
    assert generation_v3.generate("When do I enroll?", evidence)["citations"] == []
finally:
    generation_v3.ollama.chat = original_chat
print("generation assertions: ok")
