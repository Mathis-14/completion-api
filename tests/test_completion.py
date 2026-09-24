import asyncio
from app.config import Settings
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, CompletionRequest, Message
from app.services.completion import create_completion

def test_create_completion_applies_defaults(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})
    received = {}
    fake_response = Message(role="assistant", content="hello !")


    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def start(self):
            pass

        async def close(self):
            pass

        async def complete(self, messages, model, config) :
            received["messages"] = messages
            received["model"] = model
            received["config"] = config
            return fake_response

    ProviderFactory.build("fake")
    settings = Settings(
        api_keys={}, 
        default_temperature = 0.7, 
        default_max_tokens = 4096
        )
        
    request = CompletionRequest(
        provider="fake", 
        model="fake-model", 
        messages = [Message(role="user", content="Bonjour")],
        config={} 
        )

    result = asyncio.run(create_completion(request, settings))
    assert received["config"].temperature == settings.default_temperature
    assert received["config"].max_tokens == settings.default_max_tokens
    assert result is fake_response
    assert received["messages"] == request.messages
    assert received["model"] == request.model


def test_create_completion_preserves_provided_values(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})
    received = {}
    fake_response = Message(role="assistant", content="hello !")

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def start(self):
            pass

        async def close(self):
            pass

        async def complete(self, messages, model, config):
            received["messages"] = messages
            received["model"] = model
            received["config"] = config
            return fake_response

    ProviderFactory.build("fake")
    settings = Settings(
        api_keys={},
        default_temperature=0.7,
        default_max_tokens=4096,
    )
    request = CompletionRequest(
        provider="fake",
        model="fake-model",
        messages=[Message(role="user", content="Bonjour")],
        config={"temperature": 0, "max_tokens": 500},
    )

    result = asyncio.run(create_completion(request, settings))

    assert received["config"].temperature == 0
    assert received["config"].max_tokens == 500
    assert result is fake_response
    assert received["messages"] == request.messages
    assert received["model"] == request.model
