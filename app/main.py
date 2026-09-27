from fastapi import FastAPI
from app.routers import health
from app.routers import completions
from contextlib import asynccontextmanager
from app.core.orchestrator import start_providers, close_providers
from app.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await start_providers(settings)
    yield
    await close_providers()

app = FastAPI(lifespan = lifespan)
app.include_router(health.router)
app.include_router(completions.router)