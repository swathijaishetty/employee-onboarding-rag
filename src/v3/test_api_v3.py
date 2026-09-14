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
            assert payload["response_type"] == "rag"

            restored = await client.post(
                "/api/chat",
                json={
                    "question": "What about carryover?",
                    "session_id": "history-session",
                    "history": [
                        {
                            "question": "Tell me about annual leave.",
                            "search_query": "annual leave policy",
                            "answer": "Employees receive annual leave.",
                        }
                    ],
                },
            )
            assert restored.status_code == 200, restored.text
            assert len(api_v3.sessions.get("history-session").memory.turns) == 2
            assert (await client.delete("/api/sessions/test-session")).status_code == 200
            invalid = await client.post(
                "/api/chat", json={"question": " ", "session_id": "bad"}
            )
            assert invalid.status_code == 422

    asyncio.run(run_checks())
finally:
    api_v3.answer_question = original


async def run_casual_check():
    transport = httpx.ASGITransport(app=api_v3.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"question": "Hello!", "session_id": "casual-session"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["response_type"] == "greeting"
        assert payload["citations"] == []


asyncio.run(run_casual_check())
print("api assertions: ok")
