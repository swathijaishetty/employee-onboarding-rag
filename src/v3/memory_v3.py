"""Bounded session memory and conservative follow-up query rewriting."""

from collections import deque
from dataclasses import dataclass
import re

from .config_v3 import SETTINGS, Settings
from .model_client_v3 import chat


@dataclass(frozen=True)
class Turn:
    question: str
    search_query: str
    answer: str


class SessionMemory:
    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self._turns: deque[Turn] = deque(maxlen=settings.memory_turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add(self, question: str, search_query: str, answer: str) -> None:
        self._turns.append(Turn(question.strip(), search_query.strip(), answer.strip()))

    def clear(self) -> None:
        self._turns.clear()

    def resolve(self, question: str) -> str:
        question = question.strip()
        if not self._turns or not self._is_follow_up(question):
            return question
        history = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer[:500]}" for turn in self._turns
        )
        prompt = f"""Rewrite the latest question into one self-contained search query.
Use conversation history only to resolve references or omitted subjects. Preserve
the user's constraints and do not answer the question. Return the query only.

History:
{history}

Latest question: {question}
"""
        try:
            response = chat(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            rewritten = response["message"]["content"].strip().splitlines()[0]
            if 3 <= len(rewritten) <= 500:
                return rewritten
        except Exception:
            pass
        return f"{self._turns[-1].search_query} {question}"

    @staticmethod
    def _is_follow_up(question: str) -> bool:
        tokens = set(re.findall(r"[a-z]+", question.lower()))
        references = {"it", "its", "that", "this", "they", "them", "their", "those", "these", "same", "there"}
        return bool(tokens & references) or (len(tokens) <= 6 and question.lower().startswith(("what about", "and ", "how about")))
