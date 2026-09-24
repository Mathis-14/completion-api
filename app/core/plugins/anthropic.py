from anthropic import AsyncAnthropic, omit
from pydantic import SecretStr

from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import CompletionConfig, Message


@ProviderFactory.register("anthropic")
class AnthropicProvider(LLMProvider):

    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None

    async def start(self, api_key: SecretStr) -> None:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=api_key.get_secret_value())


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
                "Anthropic provider is not started. Call start() before complete(). "
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
