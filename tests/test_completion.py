import asyncio

from app.config import Settings
from app.models import CompletionConfig, CompletionRequest, Message
from app.services.completion import create_completion


def test_create_completion_applies_defaults():
    received = {}
    fake_response = Message(role="assistant", content="hello !")


    async def fake_execute_completion(
        provider_name: str,
        messages: list[Message],
        model: str,
        config: CompletionConfig,
    ) -> Message:
        received["provider_name"] = provider_name
        received["messages"] = messages
        received["model"] = model
        received["config"] = config
        return fake_response

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

    result = asyncio.run(create_completion(request, settings, fake_execute_completion))
    assert received["config"].temperature == settings.default_temperature
    assert received["config"].max_tokens == settings.default_max_tokens
    assert result is fake_response
    assert received["provider_name"] == request.provider
    assert received["messages"] == request.messages
    assert received["model"] == request.model


def test_create_completion_preserves_provided_values():
    received = {}
    fake_response = Message(role="assistant", content="hello !")

    async def fake_execute_completion(
        provider_name: str,
        messages: list[Message],
        model: str,
        config: CompletionConfig,
    ) -> Message:
        received["provider_name"] = provider_name
        received["messages"] = messages
        received["model"] = model
        received["config"] = config
        return fake_response

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

    result = asyncio.run(create_completion(request, settings, fake_execute_completion))

    assert received["config"].temperature == 0
    assert received["config"].max_tokens == 500
    assert result is fake_response
    assert received["provider_name"] == request.provider
    assert received["messages"] == request.messages
    assert received["model"] == request.model
