import pytest
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message
from app.core.exceptions import UnknownProviderError

def test_create_returns_registered_provider(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def complete(self, messages, model, config) :
            return Message(role="assistant", content="Hello!")

    provider = ProviderFactory.create("fake")
    assert isinstance(provider, FakeProvider)

def test_create_unknown_provider_raises_unknown_provider_error(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})

    with pytest.raises(UnknownProviderError):
        ProviderFactory.create("unknown")