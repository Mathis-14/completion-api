from app.core.provider import LLMProvider
from app.core.exceptions import UnknownProviderError
from contextlib import AsyncExitStack


class ProviderFactory:
    _registry: dict[str, type[LLMProvider]] = {}
    _instances: dict[str, LLMProvider] = {}
    _stack: AsyncExitStack | None = None

    @classmethod
    def register(cls, name:str):
        def decorator(provider_cls: type[LLMProvider]) -> type[LLMProvider]:
            cls._registry[name] = provider_cls
            return provider_cls
        return decorator 

    @classmethod
    def initialize(cls, stack:AsyncExitStack) -> None:
        cls._stack = stack 

    @classmethod
    def create(cls, name:str) -> LLMProvider:
        if name not in cls._registry:
            raise UnknownProviderError(f"Unknown provider: {name}")
        if cls._stack is None:
            raise RuntimeError(
                "ProviderFactory is not initialized. Call initialize() first."
                )
        if name not in cls._instances:
            cls._instances[name] = cls._registry[name](cls._stack)
        return cls._instances[name]

    @classmethod
    def reset(cls) -> None:
        cls._instances.clear()
        cls._stack = None
        