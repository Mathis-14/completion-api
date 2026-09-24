from openai import AsyncOpenAI

from app.config import get_settings
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, Message


@ProviderFactory.register("openai")
class OpenAIProvider(LLMProvider):

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None  

    async def start(self) -> None:
        if self._client is None:
            settings = get_settings()
            api_key = settings.api_keys["openai"]
            self._client =  AsyncOpenAI(api_key=api_key.get_secret_value())


    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        finally:
            self._client = None


    async def complete(
        self, messages: list[Message], model: str, config: CompletionConfig
    ) -> Message:

        if self._client is None:
            raise RuntimeError(
                "OpenAI provider is not started. Call start() before complete(). "
            )
     
        response = await self._client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens,
        )

        openai_message = response.choices[0].message
        return Message(role="assistant", content=openai_message.content)


