from app.config import Settings
from app.core.completion_executor import CompletionExecutor
from app.models import CompletionConfig, CompletionRequest, Message


async def create_completion(
    request: CompletionRequest,
    settings: Settings,
    execute_completion: CompletionExecutor,
) -> Message:

    temperature = request.config.temperature
    max_tokens = request.config.max_tokens

    if temperature is None:
        temperature = settings.default_temperature

    if max_tokens is None:
        max_tokens = settings.default_max_tokens

    effective_config = CompletionConfig(
        temperature=temperature,
        max_tokens=max_tokens
        )

    return await execute_completion(
        request.provider,
        request.messages,
        request.model,
        effective_config,
    )
