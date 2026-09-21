import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    api_keys: dict[str, SecretStr]
    default_temperature: float = Field(default=0.7, ge=0, le=1)
    default_max_tokens: int = Field(default = 4096, gt=0)


def extract_api_keys(values: dict[str, str | None]) -> dict[str, str]:
    _registry = {}
    for key,value in values.items():
        if key.endswith("_API_KEY") and value:
            _registry[key.removesuffix("_API_KEY").lower()] = value
        
    return _registry


def get_settings() -> Settings:
    values = dotenv_values(DOTENV_PATH)
    values.update(os.environ)
    api_keys = extract_api_keys(values)
    return Settings(api_keys=api_keys)
