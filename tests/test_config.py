import os

import pytest
from pydantic import SecretStr
from app import config
from app.config import extract_api_keys, get_settings


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOTENV_PATH", tmp_path / ".env")
    working_directory = tmp_path / "other_directory"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setattr(os, "environ", {})
    return tmp_path


def test_extract_api_keys_filters_values_and_preserves_secrets():
    values = {
        "PORT": "8000",
        "MISTRAL_API_KEY": "FictiveMistralAbC123",
        "OPENAI_API_KEY": "FictiveOpenAIXyZ456",
        "ANTHROPIC_API_KEY": None,
        "EMPTY_API_KEY": "",
        "MISTRAL_API_KEY_BACKUP": "ignored",
    }

    assert extract_api_keys(values) == {
        "mistral": "FictiveMistralAbC123",
        "openai": "FictiveOpenAIXyZ456",
    }


def test_extract_api_keys_returns_empty_dict_for_empty_input():
    assert extract_api_keys({}) == {}


def test_get_settings_loads_dotenv_and_masks_secrets(isolated_config):
    (isolated_config / ".env").write_text(
        "MISTRAL_API_KEY=FictiveMistralAbC123\nPORT=8000\n",
        encoding="utf-8",
    )

    settings = get_settings()

    assert set(settings.api_keys) == {"mistral"}
    assert isinstance(settings.api_keys["mistral"], SecretStr)
    assert settings.api_keys["mistral"].get_secret_value() == "FictiveMistralAbC123"
    assert "FictiveMistralAbC123" not in repr(settings)
    assert "MISTRAL_API_KEY" not in os.environ


def test_get_settings_environment_overrides_dotenv(isolated_config, monkeypatch):
    (isolated_config / ".env").write_text(
        "MISTRAL_API_KEY=FictiveFileKey\nOPENAI_API_KEY=FictiveOpenAI\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "FictiveEnvironmentKey")

    settings = get_settings()

    assert settings.api_keys["mistral"].get_secret_value() == "FictiveEnvironmentKey"
    assert settings.api_keys["openai"].get_secret_value() == "FictiveOpenAI"


def test_get_settings_loads_environment_without_dotenv(isolated_config, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "FictiveEnvironmentKey")

    settings = get_settings()

    assert set(settings.api_keys) == {"mistral"}
    assert settings.api_keys["mistral"].get_secret_value() == "FictiveEnvironmentKey"
