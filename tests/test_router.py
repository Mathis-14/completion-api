from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.exceptions import ProviderNotConfiguredError, UnknownProviderError
from app.models import CompletionConfig, Message
from app.routers.completions import get_completion_executor, router


@pytest.fixture
def completion_api():
    app = FastAPI()
    app.include_router(router)
    execute_completion = AsyncMock(return_value=Message(role="assistant", content="hi!"))
    app.dependency_overrides[get_settings] = lambda: Settings(api_keys={})
    app.dependency_overrides[get_completion_executor] = lambda: execute_completion
    with TestClient(app) as client:
        yield client, execute_completion


def test_completion_returns_200(completion_api):
    client, execute_completion = completion_api
    response = client.post(
        "/completions",
        json={
            "provider": "fake",
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi!"}],
            "config": {"temperature": 0},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"role": "assistant", "content": "hi!"}
    execute_completion.assert_awaited_once_with(
        "fake",
        [Message(role="user", content="hi!")],
        "fake-model",
        CompletionConfig(temperature=0, max_tokens=4096),
    )


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (UnknownProviderError("Unknown provider: fake"), 400, "Unknown provider: fake"),
        (
            ProviderNotConfiguredError("Provider is not configured: fake"),
            400,
            "Provider is not configured: fake",
        ),
        (TimeoutError("Workflow is still running"), 504, "Completion timed out"),
    ],
)
def test_completion_error_is_translated(completion_api, error, status_code, detail):
    client, execute_completion = completion_api
    execute_completion.side_effect = error

    response = client.post(
        "/completions",
        json={
            "provider": "fake",
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Bonjour"}],
            "config": {},
        },
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


def test_empty_messages_returns_422(completion_api):
    client, execute_completion = completion_api
    response = client.post(
        "/completions",
        json={
            "provider": "fake",
            "model": "fake-model",
            "messages": [],
            "config": {},
        },
    )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) == 1
    assert errors[0]["loc"] == ["body", "messages"]
    assert errors[0]["type"] == "too_short"
    execute_completion.assert_not_awaited()
