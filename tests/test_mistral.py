import asyncio
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from mistralai.client.errors import SDKError
from pydantic import SecretStr

from app.core.exceptions import ProviderAuthenticationError
from app.core.plugins import mistral
from app.models import CompletionConfig, Message


@pytest.fixture
def provider_client(monkeypatch):
    respond = Mock()
    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(mistral, "httpx", SimpleNamespace(
        Client=lambda **kwargs: http_client,
        AsyncClient=lambda **kwargs: async_http_client,
    ))
    monkeypatch.setattr(mistral, "Mistral", partial(mistral.Mistral, retry_config=None))
    provider = mistral.MistralProvider()
    try:
        asyncio.run(provider.start(SecretStr("fake-key")))
        respond.assert_not_called()
        yield provider, respond
    finally:
        asyncio.run(provider.close())
        assert http_client.is_closed
        assert async_http_client.is_closed


def test_mistral_reuses_client_and_returns_messages(provider_client):
    provider, respond = provider_client
    respond.side_effect = lambda request: httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0,
        "model": "fake-model",
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    messages = [Message(role="user", content="Hello")]
    client = provider._client

    async def run():
        for _ in range(2):
            result = await provider.complete(
                messages, "fake-model", CompletionConfig(temperature=0, max_tokens=100)
            )
            assert result == Message(role="assistant", content="Hello!")
            assert provider._client is client

    asyncio.run(run())
    assert respond.call_count == 2
    request = respond.call_args.args[0]
    assert request.headers["authorization"] == "Bearer fake-key"
    body = json.loads(request.content)
    assert body["model"] == "fake-model"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 100


def test_mistral_translates_authentication_failure(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx.Response(401, json={"message": "Invalid key"})

    with pytest.raises(ProviderAuthenticationError) as caught:
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))

    assert caught.value.__cause__.status_code == 401


def test_mistral_preserves_rate_limit_error(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx.Response(429, json={"message": "Rate limit"})

    with pytest.raises(SDKError) as caught:
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))

    assert caught.value.status_code == 429
