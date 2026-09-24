
from app.config import Settings
from app.core.factory import ProviderFactory
from app.core.loader import load_plugins
from app.core.provider import LLMProvider


async def start_providers(settings: Settings) -> list[LLMProvider]:
    load_plugins()
    providers = []
    try:
        for name in settings.api_keys:
            provider = ProviderFactory.build(name)
            providers.append(provider)
            await provider.start(api_key=settings.api_keys[name])
    except BaseException:
      await close_providers(providers)
      raise
    return providers


async def close_providers(providers: list[LLMProvider]) -> None:
    errors:list[Exception] = []
    for provider in reversed(providers):
        try:
            await provider.close()
        except Exception as exc:
            errors.append(exc)
        

    ProviderFactory.clear_instances()
    providers.clear()

    if errors:
        raise ExceptionGroup("Failed to close providers", errors)
