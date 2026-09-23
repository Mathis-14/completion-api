from fastapi import FastAPI
from app.routers.completions import router
from contextlib import asynccontextmanager
from app.core.loader import load_plugins
from contextlib import AsyncExitStack
from app.core.factory import ProviderFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_plugins()
    try: 
        async with AsyncExitStack() as stack:
            ProviderFactory.initialize(stack)
            yield
    finally:
        ProviderFactory.reset()

app = FastAPI(lifespan = lifespan)
app.include_router(router)