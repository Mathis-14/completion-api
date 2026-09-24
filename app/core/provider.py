from abc import ABC,abstractmethod
from app.models import Message, CompletionConfig
from pydantic import SecretStr

class LLMProvider(ABC) :

    @abstractmethod
    async def complete(
        self, messages: list[Message], model:str, config:CompletionConfig
    ) -> Message: ...


    @abstractmethod
    async def start(self, api_key: SecretStr) -> None:
        ...
    
    @abstractmethod
    async def close(self) -> None:
        ...

