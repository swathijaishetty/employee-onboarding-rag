"""Small, bounded session memory for resolving follow-up questions."""

from collections import deque
from dataclasses import dataclass
import re

import ollama

try:
    from .config_v2 import LLM_MODEL, MAX_REWRITE_CHARS, MEMORY_TURNS
except ImportError:  # pragma: no cover
    from config_v2 import LLM_MODEL, MAX_REWRITE_CHARS, MEMORY_TURNS


MAX_MEMORY_TURNS = MEMORY_TURNS


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


class SessionMemory:
    """Keep recent turns and turn context-dependent questions into standalone ones."""

    def __init__(self, max_turns: int = MAX_MEMORY_TURNS):
        self._turns: deque[Turn] = deque(maxlen=max(1, max_turns))

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question.strip(), answer.strip()))

    def clear(self) -> None:
        self._turns.clear()

    def standalone_question(self, question: str) -> str:
        """Rewrite likely follow-ups using recent context; leave standalone queries alone."""

        question = question.strip()
        if not self._turns or not self._needs_context(question):
            return question

        history = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer[:500]}"
            for turn in self._turns
        )
        prompt = f"""Rewrite the latest employee question as a standalone search query.
Resolve pronouns and omitted subjects using the conversation history.
Keep the meaning unchanged. Return only the rewritten question, with no explanation.

Conversation history:
{history}

Latest question:
{question}
"""
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = response["message"]["content"].strip()
            if rewritten and len(rewritten) <= MAX_REWRITE_CHARS:
                return rewritten.splitlines()[0].strip()
        except Exception:
            pass

        # Retrieval still has useful context if the rewrite call is unavailable.
        return f"{self._turns[-1].question} {question}".strip()

    @staticmethod
    def _needs_context(question: str) -> bool:
        lowered = question.lower()
        tokens = re.findall(r"[a-z0-9]+", lowered)
        markers = (
            "what about", "how about", "and ", "it ", "that ", "those ",
            "this ", "same", "also", "more details", "how many",
        )
        return len(tokens) <= 8 or any(marker in lowered for marker in markers)
