from app.core.provider import LLMProvider
from app.core.exceptions import UnknownProviderError



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
        return cls._registry[name]()

    @classmethod
    def publish(cls, name:str, instance:LLMProvider) ->None:
        if name not in cls._registry:
            raise UnknownProviderError(f"Unknown provider: {name}")
        cls._instances[name] = instance

    @classmethod
    def registered_names(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def get_instance(cls, name:str) -> LLMProvider:
        if name not in cls._instances:
            raise UnknownProviderError(f"Unknown provider: {name}")
        return cls._instances[name]

    @classmethod
    def registered_instances(cls) -> list[LLMProvider]:
        return list(cls._instances.values())

    @classmethod
    def clear_instances(cls) -> None:
        cls._instances.clear()