import asyncio
from mistralai.workflows import run_worker
from app.core.lifecycle import start_providers, close_providers
from app.workflows.completion import ChatCompletionWorkflow
from app.config import get_settings

async def main() -> None:
    settings = get_settings()
    providers = await start_providers(settings)
    try: 
        await run_worker([ChatCompletionWorkflow])
    
    finally:
        await close_providers(providers)


if __name__ == "__main__":
    asyncio.run(main())