from __future__ import annotations

import requests

from backend import plex_client


class _FlakyMovie:
    def __init__(self) -> None:
        self.original_title_attempts = 0
        self.title_attempts = 0

    @property
    def originalTitle(self) -> str:
        self.original_title_attempts += 1
        raise requests.exceptions.ConnectionError("boom")

    @property
    def title(self) -> str:
        self.title_attempts += 1
        return "Fallback Title"


def test_lazy_attr_guard_skips_optional_reload_but_keeps_title_fallback(
    monkeypatch,
) -> None:
    movie = _FlakyMovie()

    monkeypatch.setattr(plex_client, "_LAZY_ATTR_GUARD_CONSECUTIVE_ERRORS", 0)
    monkeypatch.setattr(plex_client, "_LAZY_ATTR_GUARD_UNTIL_MONOTONIC", 0.0)
    monkeypatch.setattr(plex_client, "PLEX_LAZY_ATTR_GUARD_THRESHOLD", 2)
    monkeypatch.setattr(plex_client, "PLEX_LAZY_ATTR_GUARD_COOLDOWN_SECONDS", 60.0)

    assert plex_client.get_original_title(movie) is None
    assert plex_client.get_original_title(movie) is None
    assert plex_client.get_original_title(movie) is None
    assert movie.original_title_attempts == 2

    assert plex_client.get_best_search_title(movie) == "Fallback Title"
    assert movie.title_attempts == 1
