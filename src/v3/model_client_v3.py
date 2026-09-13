"""Provider-neutral model access for local Ollama and hosted Gemini."""

from functools import lru_cache

import ollama

from .config_v3 import SETTINGS, Settings


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
    response = _gemini_client().models.generate_content(
        model=kwargs.get("model") or SETTINGS.llm_model,
        contents=_gemini_prompt(kwargs.get("messages") or []),
        config=types.GenerateContentConfig(temperature=options.get("temperature", 0)),
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned no text response")
    return {"message": {"content": text}}


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
