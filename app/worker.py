import asyncio
from mistralai.workflows import run_worker
from app.core.loader import load_plugins
from app.workflows.completion import ChatCompletionWorkflow

async def main() -> None:
    load_plugins()
    await run_worker([ChatCompletionWorkflow])

if __name__ == "__main__":
    asyncio.run(main())