from typing import Literal
from pydantic import BaseModel, Field

class Message(BaseModel) :
    role: Literal["user", "assistant", "system"]
    content:str

class CompletionConfig(BaseModel) :
    temperature: float | None = Field(default = None, ge=0, le=1)
    max_tokens: int | None = Field(default = None, gt=0)

class CompletionRequest(BaseModel):
    provider:str
    model:str
    messages:list[Message] = Field(min_length=1)
    config:CompletionConfig
