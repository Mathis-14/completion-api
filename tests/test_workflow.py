import asyncio
from unittest.mock import AsyncMock

import pytest

from app.models import CompletionConfig, CompletionRejection, Message
from app.workflows import completion


@pytest.mark.parametrize(
    ("result", "expected_payload"),
    [
        (
            Message(role="assistant", content="Salut"),
            {"role": "assistant", "content": "Salut"},
        ),
        (
            CompletionRejection(code="unknown_provider", detail="Unknown provider: fake"),
            {"code": "unknown_provider", "detail": "Unknown provider: fake"},
        ),
    ],
)
def test_workflow_serializes_activity_result(monkeypatch, result, expected_payload):
    activity = AsyncMock(return_value=result)
    monkeypatch.setattr(completion, "complete_with_provider", activity)

    payload = asyncio.run(completion.ChatCompletionWorkflow().run({
        "provider_name": "fake",
        "messages": [{"role": "user", "content": "Bonjour"}],
        "model": "test-model",
        "config": {"temperature": 0, "max_tokens": 100},
    }))

    assert payload == expected_payload
    activity.assert_awaited_once_with(
        "fake",
        [Message(role="user", content="Bonjour")],
        "test-model",
        CompletionConfig(temperature=0, max_tokens=100),
    )
