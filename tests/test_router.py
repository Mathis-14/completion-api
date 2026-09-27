from fastapi.testclient import TestClient
from pydantic import SecretStr
from app.main import app
from app.config import Settings, get_settings
from app.core.factory import ProviderFactory
from app.core.exceptions import ProviderAuthenticationError
from app.core.provider import LLMProvider
from app.models import Message

def test_unknown_provider_returns_400(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    def fake_get_settings():
          return Settings(
              api_keys={},
              default_temperature=0.7,
              default_max_tokens=4096,
          )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_settings,
        fake_get_settings,
    )
    monkeypatch.setattr("app.main.get_settings", fake_get_settings)

    with TestClient(app) as client:
        response = client.post(
            "/completions",
            json={
                "provider":"unknown",
                "model": "fake_model",
                "messages": [
                    {"role": "user", "content": "Bonjour"}
                ],
                "config": {},
            }

        )
        assert response.status_code == 400
        assert response.json() == {
            "detail": "Unknown provider: unknown"
        }


def test_completion_returns_200(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def start(self, api_key: SecretStr) -> None:
            pass

        async def close(self):
            pass

        async def complete(self, messages, model, config) :
            return Message(role="assistant", content="hi!")

    def fake_get_settings():
          return Settings(
              api_keys={"fake": "fake-key"},
              default_temperature=0.7,
              default_max_tokens=4096,
          )

    monkeypatch.setitem(
        app.dependency_overrides,
        get_settings,
        fake_get_settings,
    )
    monkeypatch.setattr("app.main.get_settings", fake_get_settings)

    with TestClient(app) as client:
        response = client.post(
            "/completions",
            json={
                "provider":"fake",
                "model": "fake_model",
                "messages": [
                    {"role": "user", "content": "hi!"}
                ],
                "config": {},
            }

        )
        assert response.status_code == 200
        assert response.json() == {
         "role": "assistant",
         "content": "hi!",
     }


def test_empty_messages_returns_422(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})

    def fake_get_settings():
        return Settings(
            api_keys={},
            default_temperature=0.7,
            default_max_tokens=4096,
        )

    monkeypatch.setitem(
        app.dependency_overrides,
        get_settings,
        fake_get_settings,
    )
    monkeypatch.setattr("app.main.get_settings", fake_get_settings)

    with TestClient(app) as client:
        response = client.post(
            "/completions",
            json={
                "provider": "unknown",
                "model": "fake_model",
                "messages": [],
                "config": {},
            },
        )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) == 1
    assert errors[0]["loc"] == ["body", "messages"]
    assert errors[0]["type"] == "too_short"


def test_provider_authentication_failure_returns_500_and_closes_normally(monkeypatch):
    monkeypatch.setattr(ProviderFactory, "_registry", {})
    monkeypatch.setattr(ProviderFactory, "_instances", {})
    monkeypatch.setattr("app.core.orchestrator.load_plugins", lambda: None)
    events = []

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def start(self, api_key: SecretStr) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

        async def complete(self, messages, model, config):
            events.append("complete")
            raise ProviderAuthenticationError(
                "Authentication failed for provider: fake"
            )

    def fake_get_settings():
        return Settings(api_keys={"fake": "fake-key"})

    monkeypatch.setitem(app.dependency_overrides, get_settings, fake_get_settings)
    monkeypatch.setattr("app.main.get_settings", fake_get_settings)

    with TestClient(app) as client:
        response = client.post(
            "/completions",
            json={
                "provider": "fake",
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "config": {},
            },
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Authentication failed for provider: fake"
        }
        assert events == ["start", "complete"]

    assert events == ["start", "complete", "close"]
    assert ProviderFactory.registered_instances() == []
