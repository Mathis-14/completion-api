from abc import ABC,abstractmethod
from app.models import Message, CompletionConfig

class LLMProvider(ABC) :

    @abstractmethod
    async def complete(
        self, messages: list[Message], model:str, config:CompletionConfig
    ) -> Message: ...

