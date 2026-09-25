# completion-api

Multi-provider completion API built with FastAPI, organized into layers:
routers → services → core.

`POST /completions` starts a Mistral Workflow and waits for its result. A separate
worker runs the completion activity and selects Mistral, OpenAI or Anthropic
through the provider factory and automatic plugin discovery. The API reuses one
workflow client per process; the worker owns the LLM provider clients.

Tests use simulated workflow responses and providers, without external network
calls. Live testing has confirmed dispatch from the API to the worker and a
provider rate-limit failure after the activity exhausted its attempts.

Requirements: Python 3.14+ and uv.

Set the keys for the providers you use in a `.env` file at the project root:

- `MISTRAL_API_KEY` is required for Workflows, and also enables provider `mistral`.
- `OPENAI_API_KEY` for provider `openai`.
- `ANTHROPIC_API_KEY` for provider `anthropic`.

Provider API keys are read only from `.env`; shell environment variables are ignored for these keys.

The Workflows SDK also requires a stable deployment name. Add it to `.env`:

```dotenv
DEPLOYMENT_NAME=completion-api-local
```

Specify the provider and a compatible model in each request. Model-specific
restrictions on generation parameters, such as temperature, still apply.

An unknown or unconfigured provider returns HTTP 400. These refusals are returned
as structured workflow results and are not retried. Exceptions from the provider
call still propagate to the activity retry mechanism.

The API waits up to 120 seconds by default. Set `WORKFLOW_TIMEOUT_SECONDS` in the
API process environment to override it (greater than 0 and less than 300 seconds).
This setting is not loaded from `.env`. An expired wait or HTTP timeout returns
HTTP 504; it does not cancel the remote workflow. Other unhandled failures return
HTTP 500. In particular, provider rate-limit errors (429) currently surface as
HTTP 500 after activity attempts are exhausted; preserving that error across the
workflow boundary is still to be implemented. Anthropic responses without text
blocks currently return an empty content string.

Start the API:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Start the worker in a second terminal from the repository root, with Workflows
access configured for the same Mistral workspace:

```bash
uv run python -m app.worker
```

Interactive documentation: http://127.0.0.1:8000/docs

Run the tests:

```bash
uv run python -m pytest tests -v
```
