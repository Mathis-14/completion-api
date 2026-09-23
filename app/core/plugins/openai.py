from contextlib import AsyncExitStack

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, Message


@ProviderFactory.register("openai")
class OpenAIProvider(LLMProvider):
    def __init__(self, stack: AsyncExitStack) -> None:
        super().__init__(stack)
        self._client: AsyncOpenAI | None = None

    async def complete(
        self, messages: list[Message], model: str, config: CompletionConfig
    ) -> Message:
        async with self._client_lock:
            if self._client is None:
                settings = get_settings()
                api_key = settings.api_keys["openai"]
                self._client = await self._stack.enter_async_context(
                    AsyncOpenAI(api_key=api_key.get_secret_value())
                )

        response = await self._client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens,
        )

        openai_message = response.choices[0].message
        return Message(role="assistant", content=openai_message.content)
