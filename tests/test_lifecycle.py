from app.core import lifecycle
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
import asyncio
from app.config import Settings
from pydantic import SecretStr

def test_start_providers_starts_configured_providers(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})
    monkeypatch.setattr(lifecycle, "load_plugins", lambda: None)
    received = {}

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):

        def __init__(self) -> None:
            self.started = False

        async def start(self, api_key: SecretStr) -> None:
            received["api_key"] = api_key
            self.started = True

        async def close(self) -> None:
             pass

        async def complete(self, messages, model, config):
             raise NotImplementedError

    settings = Settings(api_keys={"fake": "fake-key"})
    providers = asyncio.run(lifecycle.start_providers(settings))

    assert len(providers) == 1
    assert providers[0].started is True
    assert providers[0] is ProviderFactory.get_instance("fake")
    assert received["api_key"] == settings.api_keys["fake"]
