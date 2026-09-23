from contextlib import AsyncExitStack

from anthropic import AsyncAnthropic, omit

from app.config import get_settings
from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, Message


@ProviderFactory.register("anthropic")
class AnthropicProvider(LLMProvider):
    def __init__(self, stack: AsyncExitStack) -> None:
        super().__init__(stack)
        self._client: AsyncAnthropic | None = None

    async def complete(
        self, messages: list[Message], model: str, config: CompletionConfig
    ) -> Message:
        async with self._client_lock:
            if self._client is None:
                settings = get_settings()
                api_key = settings.api_keys["anthropic"]
                self._client = await self._stack.enter_async_context(
                    AsyncAnthropic(api_key=api_key.get_secret_value())
                )

        system = [
            {"type": "text", "text": message.content}
            for message in messages
            if message.role == "system"
        ]
        conversation = [
            message.model_dump() for message in messages if message.role != "system"
        ]

        response = await self._client.messages.create(
            model=model,
            system=system or omit,
            messages=conversation,
            temperature=config.temperature,
            max_tokens=config.max_tokens, #max_tokens is mandatory for anthropic
        )

        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return Message(role="assistant", content=content)
