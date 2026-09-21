from app.core.provider import LLMProvider
from app.core.exceptions import UnknownProviderError


class ProviderFactory:
    _registry: dict[str, type[LLMProvider]] = {}

    @classmethod
    def register(cls, name:str):
        def decorator(provider_cls: type[LLMProvider]) -> type[LLMProvider]:
            cls._registry[name] = provider_cls
            return provider_cls
        return decorator 


    @classmethod
    def create(cls, name:str) -> LLMProvider:
        if name not in cls._registry:
            raise UnknownProviderError(f"Unknown provider: {name}")
        return cls._registry[name]()