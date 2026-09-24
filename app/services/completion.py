from app.config import Settings
from app.core.factory import ProviderFactory
from app.models import CompletionConfig, CompletionRequest, Message

async def create_completion(
    request: CompletionRequest,
    settings: Settings,
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


    provider = ProviderFactory.get_instance(request.provider)
    response = await provider.complete(request.messages, request.model, effective_config)
    return response

