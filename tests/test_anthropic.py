import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core.plugins import anthropic
from app.models import CompletionConfig, Message


@pytest.mark.parametrize("system_texts", [[], ["Keep your answer short.", "Be polite."]])
def test_anthropic_complete_returns_message(monkeypatch, system_texts):
    def fake_get_settings():
        return Settings(api_keys={"anthropic": "fake-key"})

    received = {}
    fake_response = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="Internal reasoning"),
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="text", text="Mathis!"),
    ])

    class FakeMessages:
        async def create(self, *, model, system, messages, temperature, max_tokens):
            received["model"] = model
            received["system"] = system
            received["messages"] = messages
            received["temperature"] = temperature
            received["max_tokens"] = max_tokens
            return fake_response

    @asynccontextmanager
    async def fake_anthropic(*, api_key):
        received["api_key"] = api_key
        yield SimpleNamespace(messages=FakeMessages())

    monkeypatch.setattr(anthropic, "get_settings", fake_get_settings)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", fake_anthropic)

    messages = [Message(role="system", content=text) for text in system_texts]
    messages.extend([
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!"),
        Message(role="user", content="My name is Mathis."),
    ])
    config = CompletionConfig(temperature=0, max_tokens=100)
    provider = anthropic.AnthropicProvider()
    result = asyncio.run(provider.complete(messages, "fake-model", config))

    assert received["api_key"] == "fake-key"
    assert received["model"] == "fake-model"
    if system_texts:
        assert received["system"] == [
            {"type": "text", "text": "Keep your answer short."},
            {"type": "text", "text": "Be polite."},
        ]
    else:
        assert received["system"] is anthropic.omit
    assert received["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "My name is Mathis."},
    ]
    assert received["temperature"] == 0
    assert received["max_tokens"] == 100
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Hello Mathis!"
