from app.core.factory import ProviderFactory
from app.core.provider import LLMProvider
from app.models import Message, CompletionConfig
from mistralai.client import Mistral
from pydantic import SecretStr
import httpx

#Mistral SDK does not provide a puplic method close() like anthropic or openai




@ProviderFactory.register("mistral")
class MistralProvider(LLMProvider) :

    def __init__(self) -> None:
        self._client:Mistral | None = None
        self._http_client: httpx.Client | None = None
        self._async_http_client: httpx.AsyncClient | None = None

    async def start(self, api_key: SecretStr) -> None:
        if self._client is not None:
            return
        self._http_client = httpx.Client(follow_redirects=True) 
        self._async_http_client = httpx.AsyncClient(follow_redirects=True)

        self._client = Mistral(api_key=api_key.get_secret_value(), client = self._http_client, async_client = self._async_http_client)

    async def close(self) -> None:
        async_http_client = self._async_http_client
        http_client = self._http_client

        self._client = None
        self._http_client = None
        self._async_http_client = None

        try:
            if async_http_client is not None:
                await async_http_client.aclose()
        finally:
            if http_client is not None:
                http_client.close()


    async def complete(self, messages: list[Message], model:str, config:CompletionConfig) -> Message: 

        if self._client is None:
            raise RuntimeError(
                "Mistral provider is not started. Call start() before complete(). "
            )

        response = await self._client.chat.complete_async(
            model=model, 
            messages=[message.model_dump() for message in messages], 
            temperature=config.temperature, 
            max_tokens=config.max_tokens
            )

        mistral_message = response.choices[0].message
        return Message(role="assistant", content=mistral_message.content)

        
