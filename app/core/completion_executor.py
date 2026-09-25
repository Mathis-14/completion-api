from collections.abc import Awaitable, Callable

from app.models import CompletionConfig, Message


type CompletionExecutor = Callable[
    [str, list[Message], str, CompletionConfig], Awaitable[Message]
]
