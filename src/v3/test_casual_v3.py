"""Checks that casual messages bypass retrieval and do not alter RAG memory."""

from . import main_v3
from .memory_v3 import SessionMemory


def retrieval_must_not_run(_query):
    raise AssertionError("Casual messages must not run retrieval")


original_retrieve = main_v3.retrieve
main_v3.retrieve = retrieval_must_not_run
try:
    memory = SessionMemory()
    greeting = main_v3.answer_question("Hello!", memory)
    acknowledgement = main_v3.answer_question("Thank you so much.", memory)

    assert greeting["response_type"] == "greeting"
    assert acknowledgement["response_type"] == "acknowledgement"
    assert greeting["citations"] == acknowledgement["citations"] == []
    assert memory.turns == ()
finally:
    main_v3.retrieve = original_retrieve

print("casual response assertions: ok")
