import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.config import Settings
from app.core.plugins import openai
from app.models import CompletionConfig, Message


def test_openai_complete_returns_message(monkeypatch):
    def fake_get_settings():
        return Settings(api_keys={"openai": "fake-key"})

    received = {}
    fake_message = SimpleNamespace(role="assistant", content="Hello Mathis!")
    fake_response = SimpleNamespace(choices=[SimpleNamespace(message=fake_message)])

    class FakeCompletions:
        async def create(self, *, model, messages, temperature, max_completion_tokens):
            received["model"] = model
            received["messages"] = messages
            received["temperature"] = temperature
            received["max_completion_tokens"] = max_completion_tokens
            return fake_response

    def fake_openai(*, api_key):
        received["api_key"] = api_key
        return SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()), close=AsyncMock()
        )

    monkeypatch.setattr(openai, "get_settings", fake_get_settings)
    monkeypatch.setattr(openai, "AsyncOpenAI", fake_openai)

    messages = [
        Message(role="system", content="Keep your answer short."),
        Message(role="user", content="Hello"),
    ]
    config = CompletionConfig(temperature=0, max_tokens=100)
    provider = openai.OpenAIProvider()

    async def run_completion():
        await provider.start()
        try:
            return await provider.complete(messages, "fake-model", config)
        finally:
            await provider.close()

    result = asyncio.run(run_completion())

    assert received["api_key"] == "fake-key"
    assert received["model"] == "fake-model"
    assert received["messages"] == [
        {"role": "system", "content": "Keep your answer short."},
        {"role": "user", "content": "Hello"},
    ]
    assert received["temperature"] == 0
    assert received["max_completion_tokens"] == 100
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Hello Mathis!"
