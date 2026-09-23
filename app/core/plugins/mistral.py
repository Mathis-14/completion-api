from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message, CompletionConfig
from app.config import get_settings
from mistralai.client import Mistral
from contextlib import AsyncExitStack


@ProviderFactory.register("mistral")
class MistralProvider(LLMProvider) :

    def __init__(self, stack: AsyncExitStack) -> None:
        super().__init__(stack)
        self._client:Mistral | None = None

    async def complete(self, messages: list[Message], model:str, config:CompletionConfig) -> Message: 
        async with self._client_lock:
            if self._client is None:
                settings = get_settings()
                api_key = settings.api_keys["mistral"]
                self._client = await self._stack.enter_async_context(
                    Mistral(api_key=api_key.get_secret_value())
                    )
        response = await self._client.chat.complete_async(
            model=model, 
            messages=[message.model_dump() for message in messages], 
            temperature=config.temperature, 
            max_tokens=config.max_tokens
            )

        mistral_message = response.choices[0].message
        return Message(role="assistant", content=mistral_message.content)

            
