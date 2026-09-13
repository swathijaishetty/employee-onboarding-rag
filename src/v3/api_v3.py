"""FastAPI application exposing the Version 3 RAG pipeline and web client."""

from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config_v3 import SETTINGS
from .generation_v3 import FALLBACK
from .ingest_v3 import get_collection
from .main_v3 import answer_question
from .model_client_v3 import list_models, provider_name
from .session_store_v3 import RateLimiter, SessionStore


WEB_DIRECTORY = Path(__file__).with_name("web")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
app = FastAPI(
    title="Northstar Employee Policy Assistant",
    version="3.0.0",
    description="Hybrid PDF RAG API with grounded citations and session memory.",
)
sessions = SessionStore()
rate_limiter = RateLimiter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=SETTINGS.max_question_chars)
    session_id: str = Field(min_length=8, max_length=128)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not SESSION_ID_PATTERN.fullmatch(value):
            raise ValueError("Use only letters, numbers, hyphens, and underscores")
        return value


class Citation(BaseModel):
    number: int
    source: str
    page: int
    section: str
    policy_id: str = ""


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    search_query: str
    citations: list[Citation]
    retrieval: list[dict[str, Any]] | None = None


def _configured_models() -> list[str]:
    response = list_models()
    raw_models = getattr(response, "models", None)
    if raw_models is None and isinstance(response, dict):
        raw_models = response.get("models", [])
    names: list[str] = []
    for model in raw_models or []:
        name = getattr(model, "model", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.append(str(name))
    return names


@app.get("/api/health")
async def health() -> dict:
    provider = provider_name()
    details: dict[str, Any] = {
        "status": "healthy", "version": "3.0.0", "model_provider": provider
    }
    try:
        collection = await run_in_threadpool(get_collection)
        metadata = await run_in_threadpool(
            lambda: collection.get(include=["metadatas"])["metadatas"] or []
        )
        sources = {str(item.get("source")) for item in metadata if item and item.get("source")}
        details.update(
            indexed_chunks=collection.count(),
            indexed_documents=len(sources),
            collection=SETTINGS.collection_name,
        )
    except Exception:
        details.update(status="degraded", index_error="Index unavailable")
    try:
        details.update(
            model_service="available",
            models=await run_in_threadpool(_configured_models),
        )
    except Exception:
        details.update(status="degraded", model_service="unavailable")
    details["active_sessions"] = sessions.count
    return details


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not rate_limiter.allow(request.session_id):
        raise HTTPException(429, "Too many questions; wait a minute and try again")
    session = sessions.get(request.session_id)

    def run_pipeline() -> dict:
        with session.lock:
            return answer_question(request.question, session.memory)

    try:
        result = await run_in_threadpool(run_pipeline)
    except Exception as error:
        message = str(error).lower()
        if "resource_exhausted" in message or "quota" in message or "429" in message:
            raise HTTPException(
                429, "The free model quota is exhausted; try again after it resets"
            ) from error
        if any(term in message for term in ("connect", "ollama", "gemini", "api key")):
            raise HTTPException(
                503, f"The {provider_name().title()} model service is unavailable"
            ) from error
        if "collection" in message:
            raise HTTPException(503, "The Version 3 index is unavailable; run ingestion first") from error
        raise HTTPException(500, "The assistant could not process this question") from error

    evidence = result["evidence"]
    citations = [
        Citation(
            number=number,
            source=evidence[number - 1].source,
            page=evidence[number - 1].page,
            section=evidence[number - 1].section,
            policy_id=evidence[number - 1].policy_id,
        )
        for number in result["citations"]
    ]
    retrieval = None
    if request.debug:
        retrieval = [
            {
                "rank": rank,
                "source": item.source,
                "page": item.page,
                "section": item.section,
                "semantic_distance": item.semantic_distance,
                "bm25_score": round(item.lexical_score, 4),
                "fused_score": round(item.fused_score, 6),
            }
            for rank, item in enumerate(evidence, start=1)
        ]
    return ChatResponse(
        answer=result.get("answer") or FALLBACK,
        session_id=request.session_id,
        search_query=result["search_query"],
        citations=citations,
        retrieval=retrieval,
    )


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str) -> dict:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(422, "Invalid session ID")
    return {"session_id": session_id, "cleared": sessions.clear(session_id)}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIRECTORY), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.v3.api_v3:app", host=SETTINGS.api_host, port=SETTINGS.api_port)
