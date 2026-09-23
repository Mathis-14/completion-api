from abc import ABC,abstractmethod
from app.models import Message, CompletionConfig
from contextlib import AsyncExitStack
from asyncio import Lock

class LLMProvider(ABC) :

    def __init__(self, stack: AsyncExitStack) -> None:
        self._stack = stack
        self._client_lock = Lock()
        
    @abstractmethod
    async def complete(
        self, messages: list[Message], model:str, config:CompletionConfig
    ) -> Message: ...

