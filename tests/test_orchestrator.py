import asyncio

import pytest

from app.config import Settings
from app.core import orchestrator
from app.core.exceptions import UnknownProviderError
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self):
        self.api_key = None
        self.closed = False

    async def start(self, api_key):
        self.api_key = api_key

    async def close(self):
        self.closed = True

    async def complete(self, messages, model, config):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_factory(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})
    monkeypatch.setattr(orchestrator, "load_plugins", lambda: None)


def test_start_publishes_only_registered_configured_providers(monkeypatch, caplog):
    class ConfiguredProvider(FakeProvider):
        async def start(self, api_key):
            with pytest.raises(UnknownProviderError):
                ProviderFactory.get_instance("fake")
            await super().start(api_key)

    class UnconfiguredProvider(FakeProvider):
        def __init__(self):
            raise AssertionError("A provider without a key must not be built")

    def fake_load_plugins():
        ProviderFactory.register("fake")(ConfiguredProvider)
        ProviderFactory.register("unconfigured")(UnconfiguredProvider)

    monkeypatch.setattr(orchestrator, "load_plugins", fake_load_plugins)
    settings = Settings(api_keys={"fake": "fake-key", "no-plugin": "unused-key"})
    asyncio.run(orchestrator.start_providers(settings))

    provider = ProviderFactory.get_instance("fake")
    assert ProviderFactory.registered_instances() == [provider]
    assert provider.api_key == settings.api_keys["fake"]
    assert caplog.records == []


@pytest.mark.parametrize("failure_stage", ["build", "start", "cleanup"])
def test_start_failure_is_isolated(failure_stage, caplog):
    closed = []
    startup_error = RuntimeError("simulated startup failure")
    cleanup_error = RuntimeError("simulated cleanup failure")

    class BrokenProvider(FakeProvider):
        def __init__(self):
            if failure_stage == "build":
                raise startup_error
            super().__init__()

        async def start(self, api_key):
            raise startup_error

        async def close(self):
            closed.append(self)
            if failure_stage == "cleanup":
                raise cleanup_error

    ProviderFactory.register("before")(FakeProvider)
    ProviderFactory.register("broken")(BrokenProvider)
    ProviderFactory.register("after")(FakeProvider)
    settings = Settings(
        api_keys={name: "fake-key" for name in ("before", "broken", "after")}
    )
    asyncio.run(orchestrator.start_providers(settings))

    before = ProviderFactory.get_instance("before")
    after = ProviderFactory.get_instance("after")
    assert ProviderFactory.registered_instances() == [before, after]
    assert not before.closed and not after.closed
    assert len(closed) == (0 if failure_stage == "build" else 1)
    assert "broken" in caplog.records[0].getMessage()
    assert caplog.records[0].exc_info[1] is startup_error
    if failure_stage == "cleanup":
        assert caplog.records[1].exc_info[1] is cleanup_error


def test_shutdown_closes_other_instances_despite_failure(caplog):
    close_error = RuntimeError("simulated close failure")

    class BrokenProvider(FakeProvider):
        async def close(self):
            await super().close()
            raise close_error

    ProviderFactory.register("working")(FakeProvider)
    ProviderFactory.register("broken")(BrokenProvider)
    working = ProviderFactory.build("working")
    broken = ProviderFactory.build("broken")
    ProviderFactory.publish("working", working)
    ProviderFactory.publish("broken", broken)

    asyncio.run(orchestrator.close_providers())

    assert working.closed and broken.closed
    assert ProviderFactory.registered_instances() == []
    assert caplog.records[0].exc_info[1] is close_error
