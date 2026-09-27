import asyncio
import json
from functools import partial
from unittest.mock import Mock

import httpx2
import pytest
from openai import RateLimitError
from pydantic import SecretStr

from app.core.exceptions import ProviderAuthenticationError
from app.core.plugins import openai
from app.models import CompletionConfig, Message


@pytest.fixture
def provider_client(monkeypatch):
    respond = Mock()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    monkeypatch.setattr(openai, "AsyncOpenAI", partial(
        openai.AsyncOpenAI, http_client=http_client, max_retries=0,
    ))
    provider = openai.OpenAIProvider()
    try:
        asyncio.run(provider.start(SecretStr("fake-key")))
        respond.assert_not_called()
        yield provider, respond
    finally:
        asyncio.run(provider.close())
        assert http_client.is_closed


def test_openai_reuses_client_and_returns_messages(provider_client):
    provider, respond = provider_client
    respond.side_effect = lambda request: httpx2.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0,
        "model": "fake-model",
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }],
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
    assert json.loads(request.content) == {
        "model": "fake-model", "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0, "max_completion_tokens": 100,
    }


def test_openai_translates_authentication_failure(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx2.Response(401, json={"error": {"message": "Invalid key"}})

    with pytest.raises(ProviderAuthenticationError) as caught:
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))

    assert caught.value.__cause__.status_code == 401


def test_openai_preserves_rate_limit_error(provider_client):
    provider, respond = provider_client
    respond.return_value = httpx2.Response(429, json={"error": {"message": "Rate limit"}})

    with pytest.raises(RateLimitError):
        asyncio.run(provider.complete([], "fake-model", CompletionConfig(max_tokens=10)))
