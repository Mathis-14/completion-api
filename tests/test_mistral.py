import asyncio
from types import SimpleNamespace

from pydantic import SecretStr
from app.models import Message, CompletionConfig
from app.core.plugins import mistral


def test_mistral_complete_returns_message(monkeypatch):
    messages = [
        Message(role="system", content="Keep your answer short."),
        Message(role="user", content="Hello"),
    ]
    model = "fake-model"
    config = CompletionConfig(temperature=0, max_tokens=100)
    received = {}

    fake_message = SimpleNamespace(role="assistant", content="Hello Mathis!")
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)]
    )

    class FakeChat:
        async def complete_async(self, *, model, messages, temperature, max_tokens):
            received["model"] = model
            received["messages"] = messages
            received["temperature"] = temperature
            received["max_tokens"] = max_tokens
            return fake_response

    def fake_mistral(*, api_key, client, async_client):
        received["api_key"] = api_key
        return SimpleNamespace(chat=FakeChat())

    monkeypatch.setattr(mistral, "Mistral", fake_mistral)

    provider = mistral.MistralProvider()

    async def run_completion():
        await provider.start(api_key=SecretStr("fake-key"))
        try:
            return await provider.complete(messages, model, config)
        finally:
            await provider.close()

    result = asyncio.run(run_completion())

    assert received["api_key"] == "fake-key"
    assert received["model"] == model
    assert received["messages"] == [
        {"role": "system", "content": "Keep your answer short."},
        {"role": "user", "content": "Hello"},
    ]
    assert received["temperature"] == 0
    assert received["max_tokens"] == 100
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Hello Mathis!"
