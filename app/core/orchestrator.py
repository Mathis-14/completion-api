from app.config import Settings
from app.core.factory import ProviderFactory
from app.core.loader import load_plugins
import logging



async def start_providers(settings: Settings) -> None:
    load_plugins()
    names = ProviderFactory.registered_names()
    for name in names:
        provider = None
        try:
            if name not in settings.api_keys :
                continue
            provider = ProviderFactory.build(name)
            await provider.start(api_key=settings.api_keys[name])
            ProviderFactory.publish(name, provider)
        except Exception:
            logging.exception(
                "Failed to start the provider %s", name
            )
            if provider is not None:
                try:
                    await provider.close()
                except Exception:
                    logging.exception(
                        "Failed to close the provider %s", type(provider).__name__
                    )



async def close_providers() -> None:
    providers = ProviderFactory.registered_instances()
    for provider in reversed(providers):
        try:
            await provider.close()
        except Exception:
            logging.exception(
                "Failed to close the provider %s", type(provider).__name__
            )
    ProviderFactory.clear_instances()
