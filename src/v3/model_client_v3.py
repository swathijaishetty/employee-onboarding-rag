"""Single configurable Ollama client for local and hosted execution."""

import ollama

from .config_v3 import SETTINGS, Settings


def create_client(settings: Settings = SETTINGS) -> ollama.Client:
    options = {}
    if settings.ollama_host:
        options["host"] = settings.ollama_host
    if settings.ollama_api_key:
        options["headers"] = {"Authorization": f"Bearer {settings.ollama_api_key}"}
    return ollama.Client(**options)


CLIENT = create_client()


def chat(**kwargs):
    return CLIENT.chat(**kwargs)


def embed(**kwargs):
    return CLIENT.embed(**kwargs)


def list_models():
    return CLIENT.list()
