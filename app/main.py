from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.routers import completions, health
from app.workflows.client import open_completion_executor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    async with open_completion_executor(
        settings.api_keys["mistral"],
        timeout_seconds=settings.workflow_timeout_seconds,
    ) as execute_completion:
        app.state.execute_completion = execute_completion
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(completions.router)
