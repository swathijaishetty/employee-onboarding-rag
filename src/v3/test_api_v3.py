"""API contract checks with deterministic RAG evidence."""

import asyncio

import httpx

from . import api_v3
from .models_v3 import SearchResult


def fake_answer(question, memory):
    evidence = [SearchResult("id", "18 days", "leave_policy.pdf", 1, "1. Annual Leave")]
    memory.add(question, question, "Full-time employees receive 18 days [1].")
    return {
        "answer": "Full-time employees receive 18 days [1].",
        "citations": [1],
        "search_query": question,
        "evidence": evidence,
    }


original = api_v3.answer_question
api_v3.answer_question = fake_answer
try:
    async def run_checks():
        transport = httpx.ASGITransport(app=api_v3.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"question": "How much annual leave?", "session_id": "test-session", "debug": True},
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["citations"][0]["source"] == "leave_policy.pdf"
            assert payload["retrieval"][0]["rank"] == 1
            assert (await client.delete("/api/sessions/test-session")).status_code == 200
            invalid = await client.post(
                "/api/chat", json={"question": " ", "session_id": "bad"}
            )
            assert invalid.status_code == 422

    asyncio.run(run_checks())
finally:
    api_v3.answer_question = original
print("api assertions: ok")
