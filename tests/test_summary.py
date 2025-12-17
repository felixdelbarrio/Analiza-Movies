import pandas as pd
import pytest

from backend import summary


# ---------------------------------------------------------
# Tests para _sum_size (helper interno)
# ---------------------------------------------------------


def test_sum_size_basic() -> None:
    df = pd.DataFrame(
        {
            summary.FILE_SIZE_COL: [1.0, 2.5, None],
            "other": [1, 2, 3],
        }
    )

    total = summary._sum_size(df)
    # 1.0 + 2.5 = 3.5
    assert isinstance(total, float)
    assert abs(total - 3.5) < 1e-6


def test_sum_size_with_mask() -> None:
    df = pd.DataFrame(
        {
            summary.FILE_SIZE_COL: [1.0, 2.0, 3.0],
            summary.DECISION_COL: ["KEEP", "DELETE", "MAYBE"],
        }
    )

    mask = df[summary.DECISION_COL].isin(["DELETE", "MAYBE"])
    total = summary._sum_size(df, mask)

    # DELETE (2.0) + MAYBE (3.0) = 5.0
    assert isinstance(total, float)
    assert abs(total - 5.0) < 1e-6


def test_sum_size_no_column() -> None:
    df = pd.DataFrame({"other": [1, 2, 3]})
    total = summary._sum_size(df)
    assert total is None


def test_sum_size_mask_misaligned() -> None:
    # Fuerza la rama defensiva de error al aplicar la máscara
    df = pd.DataFrame(
        {
            summary.FILE_SIZE_COL: [1.0, 2.0, 3.0],
            "x": [10, 20, 30],
        }
    )
    # Máscara con longitud distinta → .loc(mask) lanzará
    bad_mask = pd.Series([True, False], name="x")

    total = summary._sum_size(df, bad_mask)
    assert total is None


# ---------------------------------------------------------
# Tests para compute_summary
# ---------------------------------------------------------


def test_compute_summary_basic(monkeypatch) -> None:
    # DF con decisiones y tamaños
    df = pd.DataFrame(
        {
            summary.DECISION_COL: ["KEEP", "DELETE", "MAYBE", "DELETE"],
            summary.FILE_SIZE_COL: [1.0, 2.0, 3.0, 4.0],
            "imdb_rating": [7.0, 6.0, 5.0, 8.0],
        }
    )

    # Hacemos deterministas las funciones de stats (parcheando en summary)
    monkeypatch.setattr(
        summary,
        "compute_global_imdb_mean_from_df",
        lambda df_: 7.5,
    )
    monkeypatch.setattr(
        summary,
        "get_global_imdb_mean_from_cache",
        lambda: 6.5,
    )

    result = summary.compute_summary(df)

    # Total de filas
    assert result["total_count"] == 4
    # Tamaño total
    assert abs(result["total_size_gb"] - (1.0 + 2.0 + 3.0 + 4.0)) < 1e-6

    # Conteos por decisión
    assert result["keep_count"] == 1
    assert result["delete_count"] == 2
    assert result["maybe_count"] == 1
    assert result["dm_count"] == 3  # DELETE + MAYBE

    # Tamaños por decisión
    assert abs(result["keep_size_gb"] - 1.0) < 1e-6
    assert abs(result["delete_size_gb"] - (2.0 + 4.0)) < 1e-6
    assert abs(result["maybe_size_gb"] - 3.0) < 1e-6
    assert abs(result["dm_size_gb"] - (2.0 + 3.0 + 4.0)) < 1e-6

    # Medias IMDb (las que hemos parcheado)
    assert result["imdb_mean_df"] == 7.5
    assert result["imdb_mean_cache"] == 6.5


def test_compute_summary_without_decision_column(monkeypatch) -> None:
    # Sin columna 'decision' → se devuelven conteos 0
    df = pd.DataFrame(
        {
            summary.FILE_SIZE_COL: [1.0, 2.0, 3.0],
            "imdb_rating": [7.0, 6.0, 5.0],
        }
    )

    monkeypatch.setattr(
        summary,
        "compute_global_imdb_mean_from_df",
        lambda df_: 7.0,
    )
    monkeypatch.setattr(
        summary,
        "get_global_imdb_mean_from_cache",
        lambda: 6.0,
    )

    result = summary.compute_summary(df)

    assert result["total_count"] == 3
    assert abs(result["total_size_gb"] - 6.0) < 1e-6

    # Sin columna 'decision' → todos a 0/None
    assert result["keep_count"] == 0
    assert result["keep_size_gb"] is None
    assert result["delete_count"] == 0
    assert result["delete_size_gb"] is None
    assert result["maybe_count"] == 0
    assert result["maybe_size_gb"] is None
    assert result["dm_count"] == 0
    assert result["dm_size_gb"] is None

    assert result["imdb_mean_df"] == 7.0
    assert result["imdb_mean_cache"] == 6.0


def test_compute_summary_empty_dataframe(monkeypatch) -> None:
    # DataFrame vacío pero con columnas esperadas
    df = pd.DataFrame(
        columns=[summary.DECISION_COL, summary.FILE_SIZE_COL, "imdb_rating"]
    )

    monkeypatch.setattr(
        summary,
        "compute_global_imdb_mean_from_df",
        lambda df_: None,
    )
    monkeypatch.setattr(
        summary,
        "get_global_imdb_mean_from_cache",
        lambda: 6.0,
    )

    result = summary.compute_summary(df)

    assert result["total_count"] == 0
    # Suma sobre serie vacía → 0.0
    assert result["total_size_gb"] == 0.0

    assert result["keep_count"] == 0
    assert result["delete_count"] == 0
    assert result["maybe_count"] == 0
    assert result["dm_count"] == 0

    assert result["keep_size_gb"] == 0.0
    assert result["delete_size_gb"] == 0.0
    assert result["maybe_size_gb"] == 0.0
    assert result["dm_size_gb"] == 0.0

    assert result["imdb_mean_df"] is None
    assert result["imdb_mean_cache"] == 6.0


def test_compute_summary_invalid_type_raises() -> None:
    # Debe lanzar TypeError si df_all no es un DataFrame
    with pytest.raises(TypeError):
        summary.compute_summary([])  # type: ignore[arg-type]