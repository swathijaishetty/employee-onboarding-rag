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

        previous_turn = self._turns[-1]
        history = (
            f"User: {previous_turn.question}\n"
            f"Assistant: {previous_turn.answer[:MAX_REWRITE_CHARS]}"
        )
        prompt = f"""Rewrite the latest employee question as a standalone search query.
Resolve pronouns and omitted subjects using the immediately preceding turn.
Replace only the ambiguous reference. Preserve the latest question's form and do
not introduce a request for quantities or facts it did not ask for.
Keep the meaning unchanged. Return only the rewritten question, with no explanation.

Immediately preceding turn:
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
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        referential_words = {
            "it", "its", "that", "those", "this", "these", "they", "them",
            "their", "same", "there",
        }
        if tokens & referential_words:
            return True

        # Short questions need history only when they lack a concrete topic.
        generic_words = {
            "a", "about", "and", "are", "available", "can", "days", "details",
            "do", "does", "for", "get", "how", "i", "is", "limit", "many",
            "me", "more", "please", "process", "rules", "tell", "the", "what",
            "when", "where", "which", "who", "why",
        }
        meaningful_words = tokens - generic_words
        return len(tokens) <= 8 and not meaningful_words
