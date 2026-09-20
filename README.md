# completion-api

API de completion multi-provider en cours de développement.
Modèles Pydantic, contrat `LLMProvider` et factory en place ; appels providers et routes HTTP à venir.

Prérequis : Python 3.14+ et uv.

```bash
uv sync
uv run python -m pytest tests -v
```
