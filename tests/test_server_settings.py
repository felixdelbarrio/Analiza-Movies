import os

from server.api.settings import Settings


def test_settings_from_env_cors_star(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    settings = Settings.from_env()

    assert settings.cors_allow_origins() == ["*"]
    assert settings.cors_allow_credentials is False


def test_settings_from_env_custom_origins(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    settings = Settings.from_env()

    assert settings.cors_allow_origins() == ["https://a.com", "https://b.com"]
    assert settings.cors_allow_credentials is True
