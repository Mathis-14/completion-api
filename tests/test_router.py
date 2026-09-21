from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings, get_settings
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message

def test_unknown_provider_returns_400(monkeypatch):
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

    @ProviderFactory.register("fake")
    class FakeProvider(LLMProvider):
        async def complete(self, messages, model, config) :
            return Message(role="assistant", content="hi!")

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
