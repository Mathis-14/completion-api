
from mistralai.workflows import workflow

from app.models import CompletionConfig, Message
from app.workflows.activities import complete_with_provider

@workflow.define(name="chat_completion")
class ChatCompletionWorkflow:
    @workflow.entrypoint
    async def run(
        self,
        provider_name: str,
        messages: list[Message],
        model: str,
        config:CompletionConfig,
    ) -> Message:
        return await complete_with_provider(
            provider_name, messages, model, config
        )