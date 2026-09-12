"""Offline checks for bounded memory and follow-up detection."""

from .memory_v3 import SessionMemory


memory = SessionMemory()
assert memory.resolve("How many annual leave days are provided?") == "How many annual leave days are provided?"
memory.add("Tell me about annual leave.", "annual leave policy", "Eligible employees receive annual leave.")
assert memory._is_follow_up("What about carryover?")
assert not memory._is_follow_up("What is the benefits enrollment deadline?")
memory.clear()
assert memory.turns == ()
print("memory assertions: ok")
