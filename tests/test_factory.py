import pytest
from pydantic import SecretStr
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message
from app.core.exceptions import UnknownProviderError

def test_build_returns_registered_provider(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def start(self, api_key: SecretStr) -> None:
            pass

        async def close(self):
            pass

        async def complete(self, messages, model, config) :
            return Message(role="assistant", content="Hello!")

    provider = ProviderFactory.build("fake")
    assert isinstance(provider, FakeProvider)

    # Construction alone must not make the provider available to requests.
    with pytest.raises(UnknownProviderError):
        ProviderFactory.get_instance("fake")

    ProviderFactory.publish("fake", provider)
    assert ProviderFactory.get_instance("fake") is provider
    assert ProviderFactory.registered_instances() == [provider]

    ProviderFactory.clear_instances()
    assert ProviderFactory.registered_instances() == []
    assert ProviderFactory.registered_names() == ["fake"]
    with pytest.raises(UnknownProviderError):
        ProviderFactory.get_instance("fake")

def test_build_unknown_provider_raises_unknown_provider_error(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    with pytest.raises(UnknownProviderError):
        ProviderFactory.build("unknown")
