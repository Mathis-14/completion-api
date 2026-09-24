from app.core.provider import LLMProvider
from app.core.exceptions import UnknownProviderError
from app.core.exceptions import ProviderNotConfiguredError


class ProviderFactory:
    _registry: dict[str, type[LLMProvider]] = {}
    _instances: dict[str, LLMProvider] = {}

    @classmethod
    def register(cls, name:str):
        def decorator(provider_cls: type[LLMProvider]) -> type[LLMProvider]:
            cls._registry[name] = provider_cls
            return provider_cls
        return decorator 

    @classmethod
    def build(cls, name:str) ->LLMProvider:
        if name not in cls._registry:
            raise UnknownProviderError(f"Unknown provider: {name}")
        if name in cls._instances:
            return cls._instances[name]

        cls._instances[name] = cls._registry[name]()
        return cls._instances[name]


    @classmethod
    def get_instance(cls, name:str) -> LLMProvider:
        if name not in cls._registry:
            raise UnknownProviderError(f"Unknown provider: {name}")
        if name not in cls._instances:
            raise ProviderNotConfiguredError(f"Provider is not configured: {name}")
        return cls._instances[name]


    @classmethod
    def clear_instances(cls) -> None:
        cls._instances.clear()