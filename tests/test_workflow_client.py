import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from pydantic import SecretStr, ValidationError

from app.core.exceptions import ProviderNotConfiguredError, UnknownProviderError
from app.models import CompletionConfig, Message
from app.workflows import client as workflow_client


async def run_completion(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Message:
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        async with httpx.AsyncClient(transport=transport) as async_http_client:
            client = Mistral(
                api_key="test-key",
                client=http_client,
                async_client=async_http_client,
            )
            return await workflow_client.execute_completion_workflow(
                client,
                "openai",
                [Message(role="user", content="Bonjour")],
                "test-model",
                CompletionConfig(temperature=0, max_tokens=100),
                timeout_seconds=30,
            )


def test_completion_sends_input_and_waits_for_result():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/workflows/chat_completion/execute"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["input"] == {
            "provider_name": "openai",
            "messages": [{"role": "user", "content": "Bonjour"}],
            "model": "test-model",
            "config": {"temperature": 0, "max_tokens": 100},
        }
        assert body["wait_for_result"] is True
        assert body["timeout_seconds"] == 30
        assert request.extensions["timeout"]["read"] == 300
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "result": {"role": "assistant", "content": "Salut"},
        })

    result = asyncio.run(run_completion(handle))

    assert result == Message(role="assistant", content="Salut")
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        ("COMPLETED", None),
        ("RUNNING", TimeoutError),
        ("RETRYING_AFTER_ERROR", TimeoutError),
        ("FAILED", RuntimeError),
        ("TIMED_OUT", RuntimeError),
    ],
)
def test_execution_status_is_checked_before_returning_result(status, expected_error):
    def handle(request):
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "root_execution_id": "test-execution",
            "status": status,
            "start_time": "2026-09-25T12:00:00Z",
            "end_time": None,
            "result": {"role": "assistant", "content": "Salut"},
        })

    if expected_error is None:
        assert asyncio.run(run_completion(handle)) == Message(
            role="assistant", content="Salut"
        )
    else:
        with pytest.raises(expected_error, match="test-execution"):
            asyncio.run(run_completion(handle))


def test_invalid_result_is_rejected():
    def handle(request):
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "result": {"role": "assistant"},
        })

    with pytest.raises(ValidationError):
        asyncio.run(run_completion(handle))


def test_sdk_error_is_propagated():
    def handle(request):
        return httpx.Response(503, json={"detail": "Service unavailable"})

    with pytest.raises(SDKError) as exc_info:
        asyncio.run(run_completion(handle))
    assert exc_info.value.status_code == 503


def test_network_timeout_is_translated():
    def handle(request):
        raise httpx.ReadTimeout("Timed out", request=request)

    with pytest.raises(TimeoutError, match="Workflow request timed out") as exc_info:
        asyncio.run(run_completion(handle))
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize(
    ("code", "detail", "expected_error"),
    [
        ("unknown_provider", "Unknown provider: openai", UnknownProviderError),
        (
            "provider_not_configured",
            "Provider is not configured: openai",
            ProviderNotConfiguredError,
        ),
    ],
)
def test_completion_rejection_is_translated(code, detail, expected_error):
    def handle(request):
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "result": {"code": code, "detail": detail},
        })

    with pytest.raises(expected_error, match=detail):
        asyncio.run(run_completion(handle))


def test_invalid_rejection_is_rejected():
    def handle(request):
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "result": {"code": "unexpected_error", "detail": "Invalid code"},
        })

    with pytest.raises(ValidationError):
        asyncio.run(run_completion(handle))


@pytest.mark.parametrize("failure", [None, "construction", "body", "cancellation"])
def test_http_clients_are_closed(monkeypatch, failure):
    clients = {}

    def create_client(*, api_key, client, async_client):
        clients.update(sync=client, asynchronous=async_client)
        assert api_key == "test-key"
        if failure == "construction":
            raise RuntimeError("Construction failed")
        return Mistral(api_key=api_key, client=client, async_client=async_client)

    monkeypatch.setattr(workflow_client, "Mistral", create_client)

    async def use_client():
        async with workflow_client.open_completion_executor(
            SecretStr("test-key"), timeout_seconds=30
        ) as execute_completion:
            assert callable(execute_completion)
            assert not clients["sync"].is_closed
            assert not clients["asynchronous"].is_closed
            if failure == "body":
                raise RuntimeError("Caller failed")
            if failure == "cancellation":
                raise asyncio.CancelledError()

    if failure is None:
        asyncio.run(use_client())
    else:
        error = asyncio.CancelledError if failure == "cancellation" else RuntimeError
        with pytest.raises(error):
            asyncio.run(use_client())

    assert clients["sync"].is_closed
    assert clients["asynchronous"].is_closed
