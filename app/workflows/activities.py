from datetime import timedelta

from mistralai.workflows import activity

from app.core.exceptions import ProviderNotConfiguredError, UnknownProviderError
from app.core.factory import ProviderFactory
from app.models import CompletionConfig, CompletionRejection, Message


@activity(start_to_close_timeout=timedelta(seconds=30))
async def complete_with_provider(
    provider_name: str,
    messages: list[Message],
    model: str,
    config: CompletionConfig,
) -> Message | CompletionRejection:
    try:
        provider = ProviderFactory.get_instance(provider_name)
    except UnknownProviderError as exc:
        return CompletionRejection(
            code="unknown_provider",
            detail=str(exc),
        )
    except ProviderNotConfiguredError as exc:
        return CompletionRejection(
            code="provider_not_configured",
            detail=str(exc),
        )

    return await provider.complete(messages, model, config)
