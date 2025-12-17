from __future__ import annotations

import math
import importlib.util
import pathlib
import sys

import altair as alt
import pandas as pd

# -------------------------------------------------------------------
# Carga dinámica de frontend/data_utils.py como módulo "data_utils"
# -------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_UTILS_PATH = ROOT / "frontend" / "data_utils.py"

spec = importlib.util.spec_from_file_location("data_utils", DATA_UTILS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo crear spec para {DATA_UTILS_PATH}")

data_utils = importlib.util.module_from_spec(spec)
sys.modules["data_utils"] = data_utils
spec.loader.exec_module(data_utils)  # type: ignore[assignment]


# -------------------------------------------------------------------
# Tests: safe_json_loads_single
# -------------------------------------------------------------------


def test_safe_json_loads_single_valid_json_dict() -> None:
    s = '{"a": 1, "b": 2}'
    result = data_utils.safe_json_loads_single(s)
    assert isinstance(result, dict)
    assert result["a"] == 1
    assert result["b"] == 2


def test_safe_json_loads_single_valid_json_list() -> None:
    s = '["a", "b", 3]'
    result = data_utils.safe_json_loads_single(s)
    assert isinstance(result, list)
    assert result == ["a", "b", 3]


def test_safe_json_loads_single_invalid_json() -> None:
    s = "{not valid json"
    result = data_utils.safe_json_loads_single(s)
    assert result is None


def test_safe_json_loads_single_empty_and_whitespace() -> None:
    assert data_utils.safe_json_loads_single("") is None
    assert data_utils.safe_json_loads_single("   ") is None


def test_safe_json_loads_single_already_dict_or_list() -> None:
    d = {"x": 1}
    lst = [1, 2, 3]
    assert data_utils.safe_json_loads_single(d) is d
    assert data_utils.safe_json_loads_single(lst) is lst


def test_safe_json_loads_single_other_types() -> None:
    assert data_utils.safe_json_loads_single(123) is None
    assert data_utils.safe_json_loads_single(None) is None
    assert data_utils.safe_json_loads_single(3.14) is None


# -------------------------------------------------------------------
# Tests: add_derived_columns
# -------------------------------------------------------------------


def test_add_derived_columns_basic_numeric_and_decade() -> None:
    df = pd.DataFrame(
        {
            "imdb_rating": ["7.5", "8.0", None],
            "rt_score": ["90", "N/A", "75"],
            "imdb_votes": ["10000", "5000", "bad"],
            "year": ["1999", "2005", "not-year"],
            "plex_rating": [None, "4.5", "3.0"],
            "file_size": [1024**3, 2 * 1024**3, None],
        }
    )

    out = data_utils.add_derived_columns(df)

    # No muta el original
    assert "file_size_gb" not in df.columns
    assert "decade" not in df.columns
    assert "decade_label" not in df.columns

    # Tipos numéricos
    assert pd.api.types.is_float_dtype(out["imdb_rating"])
    assert pd.api.types.is_float_dtype(out["rt_score"])
    assert pd.api.types.is_float_dtype(out["imdb_votes"])
    assert pd.api.types.is_float_dtype(out["year"])
    assert pd.api.types.is_float_dtype(out["plex_rating"])
    assert pd.api.types.is_float_dtype(out["file_size"])

    # file_size_gb
    assert "file_size_gb" in out.columns
    assert abs(out.loc[0, "file_size_gb"] - 1.0) < 1e-9
    assert abs(out.loc[1, "file_size_gb"] - 2.0) < 1e-9
    assert math.isnan(out.loc[2, "file_size_gb"])

    # decade y decade_label
    assert "decade" in out.columns
    assert "decade_label" in out.columns

    # 1999 -> 1990s, 2005 -> 2000s
    assert out.loc[0, "decade"] == 1990
    assert out.loc[0, "decade_label"] == "1990s"
    assert out.loc[1, "decade"] == 2000
    assert out.loc[1, "decade_label"] == "2000s"
    # valor no numérico -> NaN / None
    assert math.isnan(out.loc[2, "decade"])
    assert out.loc[2, "decade_label"] is None


def test_add_derived_columns_without_year_and_file_size() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "imdb_rating": [7.0, "8.5"],
        }
    )

    out = data_utils.add_derived_columns(df)

    # No debe crear file_size_gb / decade si no hay columnas base
    assert "file_size_gb" not in out.columns
    assert "decade" not in out.columns
    assert "decade_label" not in out.columns

    # imdb_rating convertido a float
    assert pd.api.types.is_float_dtype(out["imdb_rating"])
    assert out["imdb_rating"].tolist() == [7.0, 8.5]


def test_add_derived_columns_handles_invalid_numeric_values() -> None:
    df = pd.DataFrame(
        {
            "year": ["2000", "bad", None],
            "file_size": ["100", "bad", "200"],
        }
    )

    out = data_utils.add_derived_columns(df)

    # year numérico con NaNs
    assert pd.api.types.is_float_dtype(out["year"])
    assert out.loc[0, "year"] == 2000.0
    assert math.isnan(out.loc[1, "year"])
    assert math.isnan(out.loc[2, "year"])

    # file_size_gb calculado solo para valores numéricos válidos
    assert "file_size_gb" in out.columns
    assert abs(out.loc[0, "file_size_gb"] - (100 / (1024**3))) < 1e-12
    assert math.isnan(out.loc[1, "file_size_gb"])
    assert abs(out.loc[2, "file_size_gb"] - (200 / (1024**3))) < 1e-12


# -------------------------------------------------------------------
# Tests: explode_genres_from_omdb_json
# -------------------------------------------------------------------


def test_explode_genres_from_omdb_json_no_column() -> None:
    df = pd.DataFrame({"title": ["A", "B"]})
    out = data_utils.explode_genres_from_omdb_json(df)
    assert list(out.columns) == ["title", "genre"]
    assert out.empty


def test_explode_genres_from_omdb_json_basic_dict_and_str() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "omdb_json": [
                '{"Genre": "Action, Drama"}',
                {"Genre": "Comedy, Romance"},
                '{"NoGenre": "X"}',
                "invalid json",
            ],
        }
    )

    out = data_utils.explode_genres_from_omdb_json(df)

    # Esperamos filas: A -> Action, Drama; B -> Comedy, Romance
    assert "genre" in out.columns
    titles = out["title"].tolist()
    genres = out["genre"].tolist()

    assert len(out) == 4
    assert set(titles) == {"A", "B"}
    assert set(genres) == {"Action", "Drama", "Comedy", "Romance"}


def test_explode_genres_from_omdb_json_handles_empty_and_missing_genre() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "omdb_json": [
                '{"Genre": ""}',
                "{}",
                None,
            ],
        }
    )

    out = data_utils.explode_genres_from_omdb_json(df)
    # Ninguna tiene géneros válidos
    assert out.empty
    assert "genre" in out.columns


# -------------------------------------------------------------------
# Tests: build_word_counts
# -------------------------------------------------------------------


def test_build_word_counts_missing_columns() -> None:
    df = pd.DataFrame({"title": ["A", "B"]})
    out = data_utils.build_word_counts(df, ["DELETE"])
    assert out.empty
    assert list(out.columns) == ["word", "decision", "count"]

    df2 = pd.DataFrame({"decision": ["DELETE"]})
    out2 = data_utils.build_word_counts(df2, ["DELETE"])
    assert out2.empty
    assert list(out2.columns) == ["word", "decision", "count"]


def test_build_word_counts_empty_after_filter() -> None:
    df = pd.DataFrame(
        {
            "title": ["Keep Me"],
            "decision": ["KEEP"],
        }
    )
    out = data_utils.build_word_counts(df, ["DELETE"])
    assert out.empty


def test_build_word_counts_basic_counts_and_stopwords() -> None:
    df = pd.DataFrame(
        {
            "title": [
                "The House of the Dragon",
                "House of Cards",
                "La Casa de Papel",
            ],
            "decision": ["DELETE", "DELETE", "MAYBE"],
        }
    )

    out = data_utils.build_word_counts(df, ["DELETE", "MAYBE"])

    # Palabras como "the", "of", "la", "de" deben ser filtradas (stopwords)
    words = out["word"].unique().tolist()
    assert "the" not in words
    assert "of" not in words
    assert "la" not in words
    assert "de" not in words

    # Comprobamos que al menos alguna palabra "real" está
    assert "house" in words or "casa" in words

    # Ordenado por count descendente
    counts = out["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_build_word_counts_filters_short_words() -> None:
    df = pd.DataFrame(
        {
            "title": ["A B CD EFG HIJ"],
            "decision": ["DELETE"],
        }
    )
    out = data_utils.build_word_counts(df, ["DELETE"])
    # A, B, CD se filtran por longitud <= 2; EFG / HIJ se quedan
    words = out["word"].tolist()
    assert "a" not in words
    assert "b" not in words
    assert "cd" not in words
    assert "efg" in words
    assert "hij" in words


def test_build_word_counts_all_filtered_returns_empty() -> None:
    df = pd.DataFrame(
        {
            "title": ["The of la el y"],
            "decision": ["DELETE"],
        }
    )
    out = data_utils.build_word_counts(df, ["DELETE"])
    # Todas son stopwords => vacío
    assert out.empty


# -------------------------------------------------------------------
# Tests: decision_color
# -------------------------------------------------------------------


def test_decision_color_default_field() -> None:
    c = data_utils.decision_color()
    assert isinstance(c, alt.Color)

    c_dict = c.to_dict()
    # Comprueba field y tipo nominal
    assert c_dict.get("field") == "decision"
    assert c_dict.get("type") == "nominal"

    scale = c_dict.get("scale", {})
    assert scale.get("domain") == ["DELETE", "KEEP", "MAYBE", "UNKNOWN"]
    assert scale.get("range") == ["#e53935", "#43a047", "#fbc02d", "#9e9e9e"]


def test_decision_color_custom_field() -> None:
    c = data_utils.decision_color("my_field")
    c_dict = c.to_dict()
    assert c_dict.get("field") == "my_field"


# -------------------------------------------------------------------
# Tests: format_count_size
# -------------------------------------------------------------------


def test_format_count_size_without_size() -> None:
    assert data_utils.format_count_size(10, None) == "10"
    assert data_utils.format_count_size(5, float("nan")) == "5"


def test_format_count_size_with_valid_size() -> None:
    s = data_utils.format_count_size(10, 12.3456)
    assert s == "10 (12.35 GB)"


def test_format_count_size_with_convertible_string() -> None:
    s = data_utils.format_count_size(3, "7.891")
    assert s == "3 (7.89 GB)"


def test_format_count_size_with_non_convertible_size() -> None:
    s = data_utils.format_count_size(3, "not-a-number")
    assert s == "3"


def test_format_count_size_with_int_size() -> None:
    s = data_utils.format_count_size(2, 10)
    assert s == "2 (10.00 GB)"