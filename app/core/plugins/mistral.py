from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message, CompletionConfig
from app.config import get_settings
from mistralai.client import Mistral


@ProviderFactory.register("mistral")
class MistralProvider(LLMProvider) :
    async def complete(self, messages: list[Message], model:str, config:CompletionConfig) -> Message: 
        settings = get_settings()
        api_key = settings.api_keys["mistral"]

        async with Mistral(api_key=api_key.get_secret_value()) as client:
            response = await client.chat.complete_async(
                model=model, 
                messages=[message.model_dump() for message in messages], 
                temperature=config.temperature, 
                max_tokens=config.max_tokens
                )

            mistral_message = response.choices[0].message
            return Message(role="assistant", content=mistral_message.content)

            