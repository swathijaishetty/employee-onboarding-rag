"""Offline smoke checks for bounded follow-up session memory."""

try:
    from . import memory_v2
    from .memory_v2 import SessionMemory
except ImportError:  # pragma: no cover
    import memory_v2
    from memory_v2 import SessionMemory


def fake_chat(*, model, messages):
    return {"message": {"content": "How many annual leave days are available?"}}


original_chat = memory_v2.ollama.chat
memory_v2.ollama.chat = fake_chat
try:
    memory = SessionMemory(max_turns=2)
    memory.add_turn("Tell me about annual leave.", "Full-time employees receive 18 days.")
    assert memory.standalone_question("What about sick leave?") == (
        "What about sick leave?"
    )
    assert memory.standalone_question("How many are available?") == (
        "How many annual leave days are available?"
    )
    memory.add_turn("What about sick leave?", "Full-time employees receive 10 days.")
    memory.add_turn("What about benefits?", "Enrollment is required within 30 days.")
    assert len(memory.turns) == 2
    memory.clear()
    assert not memory.turns
finally:
    memory_v2.ollama.chat = original_chat

print("session memory assertions: ok")
