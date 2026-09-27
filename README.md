# completion-api

Multi-provider completion API built with FastAPI, organized into layers:
routers → services → core.

`POST /completions` supports Mistral, OpenAI and Anthropic through the provider
factory and automatic plugin discovery. Provider clients are initialized once
at startup and reused across requests. Tests use fake providers and real SDKs
with simulated HTTP responses, without external network calls.

Requirements: Python 3.14+ and uv.

Set the keys for the providers you use in a `.env` file at the project root:

- `MISTRAL_API_KEY` for provider `mistral`.
- `OPENAI_API_KEY` for provider `openai`.
- `ANTHROPIC_API_KEY` for provider `anthropic`.

Provider API keys are read only from `.env`; shell environment variables are ignored for these keys.

Specify the provider and a compatible model in each request. Model-specific
restrictions on generation parameters, such as temperature, still apply.

Plugins without a configured key are skipped silently. Startup failures are
logged per provider without preventing other providers from starting. A provider
is available to requests only after its initialization succeeds.

An unknown or unavailable provider returns HTTP 400. A provider authentication
failure returns HTTP 500 with a provider-specific message. Keys are not validated
remotely at startup. Other SDK failures remain unhandled and also result in
HTTP 500; upstream status codes such as 429 are not preserved by the router.

The Anthropic plugin does not forward temperature. Anthropic responses without
text blocks currently return an empty content string.

Normal shutdown attempts to close every provider and clears the stored instances.
Cleanup when the lifespan exits exceptionally or startup is interrupted remains
to be addressed.

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
