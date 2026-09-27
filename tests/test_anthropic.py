import asyncio
import json
from functools import partial
from unittest.mock import Mock

import httpx2
import pytest
from anthropic import RateLimitError
from pydantic import SecretStr

from app.core.exceptions import ProviderAuthenticationError
from app.core.plugins import anthropic
from app.models import CompletionConfig, Message


@pytest.fixture
def provider_client(monkeypatch):
    respond = Mock()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    monkeypatch.setattr(anthropic, "AsyncAnthropic", partial(
        anthropic.AsyncAnthropic, http_client=http_client, max_retries=0,
    ))
    provider = anthropic.AnthropicProvider()
    try:
        asyncio.run(provider.start(SecretStr("fake-key")))
        respond.assert_not_called()
        yield provider, respond
    finally:
        asyncio.run(provider.close())
        assert http_client.is_closed


@pytest.mark.parametrize("system_texts", [[], ["Keep your answer short.", "Be polite."]])
def test_anthropic_reuses_client_and_returns_text(provider_client, system_texts):
    provider, respond = provider_client
    respond.side_effect = lambda request: httpx2.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "fake-model",
        "content": [
            {"type": "thinking", "thinking": "Internal reasoning", "signature": "test"},
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "Mathis!"},
        ],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    messages = [Message(role="system", content=text) for text in system_texts]
    messages.append(Message(role="user", content="Hello"))
    client = provider._client

    async def run():
        for _ in range(2):
            result = await provider.complete(
                messages, "fake-model", CompletionConfig(temperature=0, max_tokens=100)
            )
            assert result == Message(role="assistant", content="Hello Mathis!")
            assert provider._client is client

    asyncio.run(run())
    assert respond.call_count == 2
    request = respond.call_args.args[0]
    assert request.headers["x-api-key"] == "fake-key"
    body = json.loads(request.content)
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["model"] == "fake-model"
    assert body["max_tokens"] == 100
    assert "temperature" not in body
    assert body.get("system", []) == [{"type": "text", "text": text} for text in system_texts]


def test_anthropic_translates_authentication_failure(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx2.Response(401, json={
        "type": "error", "error": {"type": "authentication_error", "message": "Invalid key"},
    })

    with pytest.raises(ProviderAuthenticationError) as caught:
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))

    assert caught.value.__cause__.status_code == 401


def test_anthropic_preserves_rate_limit_error(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx2.Response(429, json={
        "type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit"},
    })

    with pytest.raises(RateLimitError):
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))
