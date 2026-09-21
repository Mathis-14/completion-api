from fastapi import FastAPI
from app.routers.completions import router
from contextlib import asynccontextmanager
from app.core.loader import load_plugins

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_plugins()
    yield

app = FastAPI(lifespan = lifespan)
app.include_router(router)