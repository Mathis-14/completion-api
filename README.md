# completion-api

Multi-provider completion API built with FastAPI, organized into layers:
routers → services → core.

`POST /completions`, configuration, the provider factory and plugin discovery
are in place. Live provider calls remain to be implemented;
tests use a fake provider.

Requirements: Python 3.14+ and uv.

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
