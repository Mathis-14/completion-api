import asyncio
from unittest.mock import create_autospec

import pytest
from temporalio.testing import ActivityEnvironment

from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, CompletionRejection, Message
from app.workflows.activities import complete_with_provider


@pytest.fixture
def providers(monkeypatch):
    instances = {
        name: create_autospec(LLMProvider, instance=True)
        for name in ("openai", "anthropic")
    }
    for name, provider in instances.items():
        provider.complete.return_value = Message(role="assistant", content=name)
    monkeypatch.setattr(ProviderFactory, "_registry", dict.fromkeys(instances, LLMProvider))
    monkeypatch.setattr(ProviderFactory, "_instances", instances)
    return instances


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
def test_activity_uses_requested_provider_instance(providers, provider_name):
    messages = [Message(role="user", content="Bonjour")]
    config = CompletionConfig(temperature=0, max_tokens=100)
    result = asyncio.run(ActivityEnvironment().run(
        complete_with_provider, provider_name, messages, "test-model", config
    ))

    assert result == Message(role="assistant", content=provider_name)
    providers[provider_name].complete.assert_awaited_once_with(messages, "test-model", config)
    for name, provider in providers.items():
        if name != provider_name:
            provider.complete.assert_not_awaited()


@pytest.mark.parametrize(
    ("registered", "code", "detail"),
    [
        (False, "unknown_provider", "Unknown provider: fake"),
        (True, "provider_not_configured", "Provider is not configured: fake"),
    ],
)
def test_activity_returns_provider_rejection(monkeypatch, registered, code, detail):
    monkeypatch.setattr(ProviderFactory, "_registry", {"fake": LLMProvider} if registered else {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    result = asyncio.run(ActivityEnvironment().run(
        complete_with_provider,
        "fake",
        [Message(role="user", content="Bonjour")],
        "test-model",
        CompletionConfig(),
    ))

    assert result == CompletionRejection(code=code, detail=detail)


def test_provider_failure_propagates_for_activity_retries(providers):
    error = RuntimeError("Provider unavailable")
    providers["openai"].complete.side_effect = error

    with pytest.raises(RuntimeError, match="Provider unavailable") as exc_info:
        asyncio.run(ActivityEnvironment().run(
            complete_with_provider,
            "openai",
            [Message(role="user", content="Bonjour")],
            "test-model",
            CompletionConfig(),
        ))

    assert exc_info.value is error
