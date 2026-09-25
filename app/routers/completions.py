from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import Settings, get_settings
from app.core.completion_executor import CompletionExecutor
from app.core.exceptions import ProviderNotConfiguredError, UnknownProviderError
from app.models import CompletionRequest, Message
from app.services.completion import create_completion

router = APIRouter()


def get_completion_executor(request: Request) -> CompletionExecutor:
    return request.app.state.execute_completion


@router.post("/completions", response_model=Message)
async def complete(
    request: CompletionRequest,
    settings: Settings = Depends(get_settings),
    execute_completion: CompletionExecutor = Depends(get_completion_executor),
) -> Message:

    try:
        response = await create_completion(request, settings, execute_completion)
        return response
    except (UnknownProviderError, ProviderNotConfiguredError) as exc:
        raise HTTPException(
            status_code = 400,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Completion timed out",
        ) from exc
