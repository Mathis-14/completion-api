from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message

@ProviderFactory.register("mistral")
class MistralProvider(LLMProvider) :
    async def complete(self, messages, model, config) -> Message: ...

