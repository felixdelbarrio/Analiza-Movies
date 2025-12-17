# tests/test_decision_logic.py
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend import decision_logic


# -------------------------------------------------------------------
# Tests para _normalize_title (helper interno)
# -------------------------------------------------------------------


def test_normalize_title_basic_and_edge_cases() -> None:
    f = decision_logic._normalize_title  # type: ignore[attr-defined]

    assert f("The Matrix!") == "the matrix"
    assert f("  Hello   WORLD!! ") == "hello world"
    assert f("Spider-Man: Homecoming") == "spider man homecoming"
    assert f("") == ""
    assert f(None) == ""  # type: ignore[arg-type]


# -------------------------------------------------------------------
# Helpers para fake logger
# -------------------------------------------------------------------


class FakeLogger:
    def __init__(self) -> None:
        self.debug_msgs: list[str] = []
        self.info_msgs: list[str] = []
        self.warn_msgs: list[str] = []
        self.error_msgs: list[str] = []

    def debug(self, msg: object, *, always: bool = False) -> None:  # noqa: ARG002
        self.debug_msgs.append(str(msg))

    def info(self, msg: object, *, always: bool = False) -> None:  # noqa: ARG002
        self.info_msgs.append(str(msg))

    def warning(self, msg: object, *, always: bool = False) -> None:  # noqa: ARG002
        self.warn_msgs.append(str(msg))

    def error(self, msg: object, *, always: bool = False) -> None:  # noqa: ARG002
        self.error_msgs.append(str(msg))


@pytest.fixture
def fake_logger(monkeypatch: pytest.MonkeyPatch) -> FakeLogger:
    logger = FakeLogger()
    monkeypatch.setattr(decision_logic, "_logger", logger)
    return logger


# -------------------------------------------------------------------
# Tests para detect_misidentified
# -------------------------------------------------------------------


def test_detect_misidentified_no_omdb_data_returns_empty() -> None:
    hint = decision_logic.detect_misidentified(
        plex_title="Whatever",
        plex_year=2000,
        omdb_data=None,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    assert hint == ""


def test_detect_misidentified_title_mismatch_triggers_hint(
    fake_logger: FakeLogger,
) -> None:
    # Forzamos un umbral de votos bajo para no interferir con otros criterios
    # (por si cambian valores en config en el futuro)
    # En este test NO usamos votos, así que sólo nos interesa el título.
    omdb_data = {
        "Title": "Completely Different Film",
        "Year": "2000",
    }

    hint = decision_logic.detect_misidentified(
        plex_title="Another Movie Totally Unrelated",
        plex_year=2000,
        omdb_data=omdb_data,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )

    assert "Title mismatch" in hint
    # Debe haberse logueado debug con la similaridad de títulos
    assert any("Title similarity for" in msg for msg in fake_logger.debug_msgs)


def test_detect_misidentified_title_contains_no_hint() -> None:
    # Cuando un título contiene al otro, NO debería considerarse mismatch
    omdb_data = {
        "Title": "The Lord of the Rings",
        "Year": "2001",
    }
    hint = decision_logic.detect_misidentified(
        plex_title="The Lord of the Rings: Extended Edition",
        plex_year=2001,
        omdb_data=omdb_data,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    assert "Title mismatch" not in hint


def test_detect_misidentified_year_mismatch_triggers_hint() -> None:
    omdb_data = {
        "Title": "Same Title",
        "Year": "1990",
    }
    hint = decision_logic.detect_misidentified(
        plex_title="Same Title",
        plex_year=2010,
        omdb_data=omdb_data,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    assert "Year mismatch" in hint
    assert "Plex=2010" in hint
    assert "OMDb=1990" in hint


def test_detect_misidentified_year_compare_error_is_caught(fake_logger: FakeLogger) -> None:
    # Fuerza una excepción en int(omdb_year) metiendo algo raro
    omdb_data = {
        "Title": "Weird Year",
        "Year": "not-a-year",
    }
    hint = decision_logic.detect_misidentified(
        plex_title="Weird Year",
        plex_year=2010,
        omdb_data=omdb_data,
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
    )
    # No debe explotar; sin año válido, no debería añadir hint de año
    assert "Year mismatch" not in hint
    # Se debería haber logueado un debug explicando el fallo
    assert any("Could not compare years" in msg for msg in fake_logger.debug_msgs)


def test_detect_misidentified_imdb_low_with_many_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ajustamos umbrales para hacer el test independiente de la config real
    monkeypatch.setattr(decision_logic, "IMDB_RATING_LOW_THRESHOLD", 4.0)
    monkeypatch.setattr(decision_logic, "IMDB_MIN_VOTES_FOR_KNOWN", 100)

    omdb_data = {"Title": "Suspect Movie", "Year": "2010"}
    hint = decision_logic.detect_misidentified(
        plex_title="Suspect Movie",
        plex_year=2010,
        omdb_data=omdb_data,
        imdb_rating=3.5,
        imdb_votes=500,
        rt_score=None,
    )

    assert "IMDb muy baja" in hint
    assert "3.5" in hint
    assert "500" in hint


def test_detect_misidentified_rt_low_with_many_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decision_logic, "RT_RATING_LOW_THRESHOLD", 20)
    monkeypatch.setattr(decision_logic, "IMDB_MIN_VOTES_FOR_KNOWN", 50)

    omdb_data = {"Title": "RT Bad", "Year": "2018"}
    hint = decision_logic.detect_misidentified(
        plex_title="RT Bad",
        plex_year=2018,
        omdb_data=omdb_data,
        imdb_rating=7.0,
        imdb_votes=100,
        rt_score=10,
    )

    assert "RT muy bajo" in hint
    assert "10%" in hint
    assert "100 votos IMDb" in hint


def test_detect_misidentified_low_votes_no_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decision_logic, "IMDB_RATING_LOW_THRESHOLD", 4.0)
    monkeypatch.setattr(decision_logic, "RT_RATING_LOW_THRESHOLD", 20)
    monkeypatch.setattr(decision_logic, "IMDB_MIN_VOTES_FOR_KNOWN", 1000)

    omdb_data = {"Title": "Low Votes", "Year": "2015"}

    # IMDb baja pero con pocos votos → no debería generar hint de "peli conocida"
    hint = decision_logic.detect_misidentified(
        plex_title="Low Votes",
        plex_year=2015,
        omdb_data=omdb_data,
        imdb_rating=3.0,
        imdb_votes=100,
        rt_score=10,
    )

    assert "IMDb muy baja" not in hint
    assert "RT muy bajo" not in hint


def test_detect_misidentified_no_issues_returns_empty() -> None:
    omdb_data = {"Title": "Clean Movie", "Year": "2020"}
    hint = decision_logic.detect_misidentified(
        plex_title="Clean Movie",
        plex_year=2020,
        omdb_data=omdb_data,
        imdb_rating=7.5,
        imdb_votes=200,
        rt_score=80,
    )
    assert hint == ""


# -------------------------------------------------------------------
# Tests para sort_filtered_rows
# -------------------------------------------------------------------


def test_sort_filtered_rows_priority_order() -> None:
    rows = [
        {
            "title": "Keep High Votes",
            "decision": "KEEP",
            "imdb_votes": 1000,
            "imdb_rating": 8.0,
            "file_size": 10,
        },
        {
            "title": "Delete Low Votes",
            "decision": "DELETE",
            "imdb_votes": 100,
            "imdb_rating": 5.0,
            "file_size": 5,
        },
        {
            "title": "Maybe Mid",
            "decision": "MAYBE",
            "imdb_votes": 300,
            "imdb_rating": 6.5,
            "file_size": 7,
        },
        {
            "title": "Unknown Stuff",
            "decision": "UNKNOWN",
            "imdb_votes": 50,
            "imdb_rating": 7.0,
            "file_size": 3,
        },
        {
            "title": "Delete High Votes",
            "decision": "DELETE",
            "imdb_votes": 2000,
            "imdb_rating": 4.5,
            "file_size": 12,
        },
    ]

    sorted_rows = decision_logic.sort_filtered_rows(rows)

    titles_in_order = [r["title"] for r in sorted_rows]

    # Primero DELETE (ordenados por votos desc), luego MAYBE, luego KEEP, luego UNKNOWN
    assert titles_in_order[0] == "Delete High Votes"
    assert titles_in_order[1] == "Delete Low Votes"
    assert titles_in_order[2] == "Maybe Mid"
    assert titles_in_order[3] == "Keep High Votes"
    assert titles_in_order[4] == "Unknown Stuff"


def test_sort_filtered_rows_break_ties_by_rating_and_size() -> None:
    rows = [
        {
            "title": "Delete Lower Rating",
            "decision": "DELETE",
            "imdb_votes": 1000,
            "imdb_rating": 5.0,
            "file_size": 5,
        },
        {
            "title": "Delete Higher Rating",
            "decision": "DELETE",
            "imdb_votes": 1000,
            "imdb_rating": 7.0,
            "file_size": 4,
        },
        {
            "title": "Delete Same Rating Bigger File",
            "decision": "DELETE",
            "imdb_votes": 1000,
            "imdb_rating": 7.0,
            "file_size": 10,
        },
    ]

    sorted_rows = decision_logic.sort_filtered_rows(rows)
    titles = [r["title"] for r in sorted_rows]

    # Misma decisión y votos: se prioriza rating más alto y luego tamaño
    assert titles[0] == "Delete Same Rating Bigger File"
    assert titles[1] == "Delete Higher Rating"
    assert titles[2] == "Delete Lower Rating"


def test_sort_filtered_rows_handles_missing_fields_gracefully() -> None:
    rows = [
        {"title": "No Fields"},
        {"title": "Partial", "decision": "KEEP"},
        {"title": "Weird Types", "decision": 123, "imdb_votes": "not-int"},
    ]

    sorted_rows = decision_logic.sort_filtered_rows(rows)
    titles = [r["title"] for r in sorted_rows]

    # No debe lanzar; todos se consideran UNKNOWN con valores 0
    # El orden relativo es determinista basado en key_func
    assert set(titles) == {"No Fields", "Partial", "Weird Types"}