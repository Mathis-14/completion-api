# completion-api

Multi-provider completion API built with FastAPI, organized into layers:
routers → services → core.

`POST /completions` supports Mistral, OpenAI and Anthropic through the provider
factory and automatic plugin discovery. Tests use fake providers and SDK clients,
without network calls. Live calls have been verified with Mistral only.

Requirements: Python 3.14+ and uv.

Set the keys for the providers you use in a `.env` file at the project root:

- `MISTRAL_API_KEY` for provider `mistral`.
- `OPENAI_API_KEY` for provider `openai`.
- `ANTHROPIC_API_KEY` for provider `anthropic`.

Specify the provider and a compatible model in each request. Model-specific
restrictions on generation parameters, such as temperature, still apply.

Explicit handling of missing credentials, SDK failures and unsupported response
shapes remains to be implemented. Unhandled exceptions result in HTTP 500;
Anthropic responses without text blocks currently return an empty content string.

Start the API:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Interactive documentation: http://127.0.0.1:8000/docs

Run the tests:

```bash
uv run python -m pytest tests -v
```
