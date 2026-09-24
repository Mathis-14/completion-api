from fastapi import FastAPI
from app.routers.completions import router
from contextlib import asynccontextmanager
from app.core.lifecycle import start_providers, close_providers
from app.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    providers = await start_providers(settings)
    try: 
        yield
    finally:
        await close_providers(providers)

app = FastAPI(lifespan = lifespan)
app.include_router(router)