from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import httpx
from mistralai.client import Mistral
from mistralai.client.models import WorkflowExecutionResponse
from pydantic import SecretStr

from app.core.completion_executor import CompletionExecutor
from app.core.exceptions import ProviderNotConfiguredError, UnknownProviderError
from app.models import CompletionConfig, CompletionRejection, Message


async def execute_completion_workflow(
    client: Mistral,
    provider_name: str,
    messages: list[Message],
    model: str,
    config: CompletionConfig,
    *,
    timeout_seconds: float,
) -> Message:
    workflow_input = {
        "provider_name": provider_name,
        "messages": [message.model_dump() for message in messages],
        "model": model,
        "config": config.model_dump(),
    }
    try:
        response = await client.workflows.execute_workflow_async(
            workflow_identifier="chat_completion",
            input=workflow_input,
            wait_for_result=True,
            timeout_seconds=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise TimeoutError("Workflow request timed out") from exc
    if isinstance(response, WorkflowExecutionResponse):
        if response.status in {"RUNNING", "RETRYING_AFTER_ERROR"}:
            raise TimeoutError(f"Workflow {response.execution_id} is still running")
        if response.status != "COMPLETED":
            raise RuntimeError(
                f"Workflow {response.execution_id} returned status {response.status}"
            )
    if isinstance(response.result, dict) and "code" in response.result:
        rejection = CompletionRejection.model_validate(response.result)
        if rejection.code == "unknown_provider":
            raise UnknownProviderError(rejection.detail)
        raise ProviderNotConfiguredError(rejection.detail)

    return Message.model_validate(response.result)


@asynccontextmanager
async def open_completion_executor(
    api_key: SecretStr,
    *,
    timeout_seconds: float,
) -> AsyncIterator[CompletionExecutor]:
    with httpx.Client(follow_redirects=True) as http_client:
        async with httpx.AsyncClient(follow_redirects=True) as async_http_client:
            client = Mistral(
                api_key=api_key.get_secret_value(),
                client=http_client,
                async_client=async_http_client,
            )
            yield partial(
                execute_completion_workflow,
                client,
                timeout_seconds=timeout_seconds,
            )
