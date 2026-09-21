from fastapi import APIRouter, Depends, HTTPException
from app.config import Settings, get_settings
from app.models import CompletionRequest, Message
from app.services.completion import create_completion
from app.core.exceptions import UnknownProviderError

router = APIRouter()

@router.post("/completions", response_model=Message)
async def complete(
    request: CompletionRequest,
    settings: Settings = Depends(get_settings),
) -> Message:

    try:
        response = await create_completion(request, settings)
        return response
    except UnknownProviderError as exc:
        raise HTTPException(
            status_code = 400,
            detail=str(exc),
        ) from exc


