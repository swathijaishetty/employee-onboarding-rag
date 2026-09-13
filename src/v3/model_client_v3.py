"""Provider-neutral model access for local Ollama and hosted Gemini."""

from functools import lru_cache
import logging
import re
import time

import ollama

from .config_v3 import SETTINGS, Settings


LOGGER = logging.getLogger(__name__)


def create_client(settings: Settings = SETTINGS) -> ollama.Client:
    options = {}
    if settings.ollama_host:
        options["host"] = settings.ollama_host
    if settings.ollama_api_key:
        options["headers"] = {"Authorization": f"Bearer {settings.ollama_api_key}"}
    return ollama.Client(**options)


OLLAMA_CLIENT = create_client()


@lru_cache(maxsize=1)
def _gemini_client():
    if not SETTINGS.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when V3_MODEL_PROVIDER=gemini")
    from google import genai

    return genai.Client(api_key=SETTINGS.gemini_api_key)


def _gemini_prompt(messages: list[dict]) -> str:
    return "\n\n".join(
        f"{str(message.get('role', 'user')).upper()}: {message.get('content', '')}"
        for message in messages
    )


def chat(**kwargs):
    if SETTINGS.model_provider == "ollama":
        return OLLAMA_CLIENT.chat(**kwargs)
    from google.genai import types

    options = kwargs.get("options") or {}
    for attempt in range(SETTINGS.generation_max_retries + 1):
        try:
            response = _gemini_client().models.generate_content(
                model=kwargs.get("model") or SETTINGS.llm_model,
                contents=_gemini_prompt(kwargs.get("messages") or []),
                config=types.GenerateContentConfig(
                    temperature=options.get("temperature", 0)
                ),
            )
            break
        except Exception as error:
            delay = _generation_retry_delay(error, attempt)
            if delay is None or attempt == SETTINGS.generation_max_retries:
                raise
            LOGGER.warning(
                "Temporary Gemini failure (code=%s, status=%s); retrying in %.1fs",
                getattr(error, "code", None),
                getattr(error, "status", None),
                delay,
            )
            time.sleep(delay)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned no text response")
    return {"message": {"content": text}}


def _generation_retry_delay(error: Exception, attempt: int) -> float | None:
    code = getattr(error, "code", None)
    message = str(error)
    if code == 429 or re.search(r"429|RESOURCE_EXHAUSTED", message, re.I):
        matches = re.findall(
            r"(?:retry in\s+|retryDelay['\"\s:]+)([\d.]+)s", message, re.I
        )
        if not matches:
            return None
        suggested = max(map(float, matches)) + 1.0
        return suggested if suggested <= 60 else None
    if code in {500, 502, 503, 504}:
        return SETTINGS.generation_retry_base_seconds * (2**attempt)
    return None


def embed(**kwargs):
    if SETTINGS.model_provider == "ollama":
        kwargs.pop("task_type", None)
        return OLLAMA_CLIENT.embed(**kwargs)
    from google.genai import types

    content = kwargs.get("input")
    contents = content if isinstance(content, list) else [content]
    response = _gemini_client().models.embed_content(
        model=kwargs.get("model") or SETTINGS.embedding_model,
        contents=contents,
        config=types.EmbedContentConfig(task_type=kwargs.get("task_type")),
    )
    return {"embeddings": [item.values for item in response.embeddings]}


def list_models():
    if SETTINGS.model_provider == "ollama":
        return OLLAMA_CLIENT.list()
    _gemini_client().models.get(model=SETTINGS.llm_model)
    return {"models": [{"model": SETTINGS.llm_model}, {"model": SETTINGS.embedding_model}]}


def provider_name() -> str:
    return SETTINGS.model_provider
