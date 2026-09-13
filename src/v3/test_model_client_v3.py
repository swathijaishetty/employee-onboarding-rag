"""Offline checks for the Gemini adapter's provider-neutral response shapes."""

from types import SimpleNamespace

from . import model_client_v3


class FakeModels:
    def generate_content(self, **kwargs):
        assert kwargs["model"] == "gemini-test"
        assert "USER: hello" in kwargs["contents"]
        return SimpleNamespace(text="hello back")

    def embed_content(self, **kwargs):
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])


original_provider = model_client_v3.SETTINGS.model_provider
original_client = model_client_v3._gemini_client
try:
    object.__setattr__(model_client_v3.SETTINGS, "model_provider", "gemini")
    model_client_v3._gemini_client = lambda: SimpleNamespace(models=FakeModels())
    chat = model_client_v3.chat(
        model="gemini-test", messages=[{"role": "user", "content": "hello"}]
    )
    assert chat == {"message": {"content": "hello back"}}
    embeddings = model_client_v3.embed(
        model="embedding-test", input=["query"], task_type="RETRIEVAL_QUERY"
    )
    assert embeddings == {"embeddings": [[0.1, 0.2]]}
finally:
    object.__setattr__(model_client_v3.SETTINGS, "model_provider", original_provider)
    model_client_v3._gemini_client = original_client

print("model client assertions: ok")
