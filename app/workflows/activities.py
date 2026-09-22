from datetime import timedelta
from mistralai.workflows import activity

from app.core.factory import ProviderFactory
from app.models import CompletionConfig, Message

@activity(start_to_close_timeout=timedelta(seconds=30))
async def complete_with_provider(
    provider_name:str,
    messages:list[Message],
    model:str,
    config:CompletionConfig,
) -> Message:
    provider = ProviderFactory.create(provider_name)
    return await provider.complete(messages, model, config)

    
