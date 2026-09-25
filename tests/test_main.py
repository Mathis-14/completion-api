import json

import httpx
from fastapi.testclient import TestClient

from app import main
from app.config import Settings, get_settings
from app.core.factory import ProviderFactory
from app.workflows import client as workflow_client


def test_lifespan_reuses_workflow_client_and_closes_transports(monkeypatch):
    settings = Settings(api_keys={"mistral": "test-key"}, workflow_timeout_seconds=90)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setitem(main.app.dependency_overrides, get_settings, lambda: settings)
    requests = []
    clients = {"sync": [], "async": []}

    def handle(request):
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url.path == "/v1/workflows/chat_completion/execute"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={
            "workflow_name": "chat_completion",
            "execution_id": "test-execution",
            "result": {"role": "assistant", "content": body["input"]["provider_name"]},
        })

    transport = httpx.MockTransport(handle)
    sync_client_class = httpx.Client
    async_client_class = httpx.AsyncClient

    def make_sync_client(**kwargs):
        client = sync_client_class(transport=transport, **kwargs)
        clients["sync"].append(client)
        return client

    def make_async_client(**kwargs):
        client = async_client_class(transport=transport, **kwargs)
        clients["async"].append(client)
        return client

    def unexpected_provider_build(name):
        raise AssertionError("Providers must be built by the worker")

    monkeypatch.setattr(workflow_client.httpx, "Client", make_sync_client)
    monkeypatch.setattr(workflow_client.httpx, "AsyncClient", make_async_client)
    monkeypatch.setattr(ProviderFactory, "build", unexpected_provider_build)

    with TestClient(main.app) as client:
        for provider_name in ("openai", "anthropic"):
            response = client.post(
                "/completions",
                json={
                    "provider": provider_name,
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Bonjour"}],
                    "config": {},
                },
            )
            assert response.status_code == 200
            assert response.json() == {"role": "assistant", "content": provider_name}

        assert client.get("/health").json() == {"status": "ok"}
        assert len(requests) == 2
        for body in requests:
            assert body["timeout_seconds"] == 90
            assert body["wait_for_result"] is True
            assert body["input"]["config"] == {"temperature": 0.7, "max_tokens": 4096}
        for created_clients in clients.values():
            assert len(created_clients) == 1
            assert not created_clients[0].is_closed

    for created_clients in clients.values():
        assert created_clients[0].is_closed
