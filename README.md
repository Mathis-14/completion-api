# completion-api

Multi-provider completion API built with FastAPI, organized into layers:
routers → services → core.

`POST /completions` supports live Mistral calls through the provider factory
and automatic plugin discovery. Other providers remain to be implemented.
Tests use fake providers and a fake Mistral client, without network calls.

Requirements: Python 3.14+ and uv.

Set `MISTRAL_API_KEY` in a `.env` file at the project root. Use provider
`mistral` and a supported model such as `ministral-8b-latest` in requests.

Explicit handling of missing credentials, SDK failures and unsupported
response shapes remains to be implemented; these currently result in HTTP 500.

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
