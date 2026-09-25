
from mistralai.workflows import workflow

from app.models import CompletionConfig, CompletionRejection, Message
from app.workflows.activities import complete_with_provider

@workflow.define(name="chat_completion")
class ChatCompletionWorkflow:
    @workflow.entrypoint
    async def run(
        self,
        provider_name: str,
        messages: list[Message],
        model: str,
        config: CompletionConfig,
    ) -> Message | CompletionRejection:
        return await complete_with_provider(
            provider_name, messages, model, config
        )
