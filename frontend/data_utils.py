from __future__ import annotations

"""
data_utils.py

Utilidades de frontend para preparar datos del dashboard (Streamlit).

Incluye:
- safe_json_loads_single: parseo robusto de JSON almacenado en celdas (string/dict/list).
- add_derived_columns: columnas derivadas ligeras (numéricas, tamaños en GB, década).
- explode_genres_from_omdb_json: “explosión” por género a partir de omdb_json.
- build_word_counts: recuento de palabras en títulos filtrando por decisión.
- decision_color: paleta consistente por decisión (Altair).
- format_count_size: helper para presentar "N (X.XX GB)".

Principios
- No mutar inputs (se devuelve copia cuando aplica).
- Ser tolerante a NaN/None y tipos inesperados.
- Mantener esta capa “frontend” sin dependencias del backend.
"""

from collections import Counter
import json
import re
from typing import Any, cast, Iterable

import altair as alt
import pandas as pd


# ============================================================================
# Utilidades de JSON
# ============================================================================


def safe_json_loads_single(x: object) -> dict | list | None:
    """
    Parsea JSON de forma segura para una sola celda/campo.

    Reglas:
    - Si x ya es dict/list → se devuelve tal cual.
    - Si x es str no vacía → intenta json.loads, devuelve dict/list o None si falla.
    - En cualquier otro caso → None.

    Nota:
    - Se mantiene deliberadamente simple: NO intenta “arreglar” strings inválidos.
    """
    if isinstance(x, (dict, list)):
        return x

    if isinstance(x, str) and x.strip():
        try:
            parsed = json.loads(x)
        except Exception:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    return None


# ============================================================================
# Columnas derivadas para el dashboard
# ============================================================================

_NUMERIC_COLS: tuple[str, ...] = (
    "imdb_rating",
    "rt_score",
    "imdb_votes",
    "year",
    "plex_rating",
    "file_size",
    "metacritic_score",
)


def _parse_metacritic_value(value: object) -> float | None:
    """
    Parsea valores típicos de Metacritic:
      - "68" / "68/100" / 68
      - "N/A" -> None
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if pd.isna(cast(Any, value)):
            return None
        v = float(value)
        return v if 0.0 <= v <= 100.0 else None

    s = str(value).strip()
    if not s or s.upper() == "N/A":
        return None

    if "/" in s:
        left = s.split("/", 1)[0].strip()
        try:
            v = float(left)
        except Exception:
            return None
        return v if 0.0 <= v <= 100.0 else None

    try:
        v = float(s)
    except Exception:
        return None
    return v if 0.0 <= v <= 100.0 else None


def _extract_metacritic_from_omdb_json(raw: object) -> float | None:
    """
    Extrae Metacritic score desde omdb_json.

    Prioridad:
      1) Campo "Metascore" (string tipo "68" o "N/A")
      2) Array "Ratings": [{"Source":"Metacritic","Value":"68/100"}, ...]
    """
    data = safe_json_loads_single(raw)
    if not isinstance(data, dict):
        return None

    direct = _parse_metacritic_value(data.get("Metascore"))
    if direct is not None:
        return direct

    ratings = data.get("Ratings")
    if not isinstance(ratings, list):
        return None

    for item in ratings:
        if not isinstance(item, dict):
            continue
        src = str(item.get("Source") or "").strip().lower()
        if src == "metacritic":
            return _parse_metacritic_value(item.get("Value"))

    return None


def _to_numeric_series(values: object) -> pd.Series:
    """
    Helper de tipado: fuerza el resultado a Series para stubs de pandas.

    Nota: en runtime, pasar un Series aquí produce un Series siempre.
    """
    s = pd.to_numeric(cast(Any, values), errors="coerce")
    if isinstance(s, pd.Series):
        return s
    return pd.Series([s])


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas derivadas ligeras: tamaños, década, etc.

    - No modifica el DataFrame original: siempre devuelve una copia.
    - Convierte a numérico ciertas columnas si existen (errors="coerce").
    - Crea:
        * file_size_gb (si existe file_size en bytes)
        * decade y decade_label (si existe year)
        * metacritic_score (si existe omdb_json)

    Returns:
        DataFrame con columnas derivadas añadidas.
    """
    df = df.copy()

    if "omdb_json" in df.columns and "metacritic_score" not in df.columns:
        df["metacritic_score"] = df["omdb_json"].apply(_extract_metacritic_from_omdb_json)

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "file_size" in df.columns:
        file_size_num = _to_numeric_series(df["file_size"])
        df["file_size_gb"] = file_size_num.astype("float64") / (1024**3)

    if "year" in df.columns:
        year_num = _to_numeric_series(df["year"])
        decade = (year_num // 10) * 10
        df["decade"] = decade

        def _format_decade(val: object) -> str | None:
            if val is None or pd.isna(cast(Any, val)):
                return None
            if not isinstance(val, (int, float)):
                return None
            return f"{int(val)}s"

        df["decade_label"] = df["decade"].apply(_format_decade)

    return df


# ============================================================================
# Explosión de géneros desde omdb_json
# ============================================================================


def explode_genres_from_omdb_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un DataFrame “exploded” por género usando la columna omdb_json.

    - Si no existe la columna omdb_json, devuelve un DataFrame vacío con
      las mismas columnas + 'genre'.
    - Si omdb_json no es parseable o no tiene 'Genre', se ignora esa fila.

    Returns:
        DataFrame con una fila por (película, género) y columna 'genre'.
    """
    if "omdb_json" not in df.columns:
        return pd.DataFrame(columns=[*df.columns, "genre"])

    df_g = df.copy().reset_index(drop=True)

    def _extract_genres(raw: object) -> list[str]:
        data = safe_json_loads_single(raw)
        if not isinstance(data, dict):
            return []
        g = data.get("Genre")
        if not g:
            return []
        return [x.strip() for x in str(g).split(",") if x.strip()]

    df_g["genre_list"] = df_g["omdb_json"].apply(_extract_genres)
    df_g = df_g.explode("genre_list").reset_index(drop=True)
    df_g = df_g.rename(columns={"genre_list": "genre"})

    mask = (df_g["genre"].notna() & (df_g["genre"] != "")).astype(bool)
    filtered = df_g.loc[mask, :]
    return pd.DataFrame(filtered).copy()


# ============================================================================
# Recuento de palabras en títulos por decisión
# ============================================================================

_STOPWORDS = frozenset(
    {
        "the",
        "of",
        "la",
        "el",
        "de",
        "y",
        "a",
        "en",
        "los",
        "las",
        "un",
        "una",
        "and",
        "to",
        "for",
        "con",
        "del",
        "le",
        "les",
        "die",
        "der",
        "das",
    }
)

_WORD_SPLIT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def build_word_counts(df: pd.DataFrame, decisions: Iterable[str]) -> pd.DataFrame:
    """
    Construye un DataFrame con recuento de palabras en títulos.

    Reglas:
    - Filtra por las decisiones indicadas (DELETE/MAYBE, etc.).
    - Elimina stopwords básicas.
    - Elimina palabras de longitud <= 2.
    - Limpia puntuación (se sustituye por espacios).

    Returns:
        DataFrame con columnas: word, decision, count (ordenado desc por count).
    """
    if "decision" not in df.columns or "title" not in df.columns:
        return pd.DataFrame(columns=["word", "decision", "count"])

    decisions_set = set(decisions)
    mask = df["decision"].isin(decisions_set)
    df2 = df.loc[mask].copy()

    if df2.empty:
        return pd.DataFrame(columns=["word", "decision", "count"])

    rows: list[dict[str, object]] = []

    for dec, sub in df2.groupby("decision"):
        dec_str = str(dec)
        words: list[str] = []

        for title_obj in sub["title"].dropna():
            title = str(title_obj)
            t_clean = _WORD_SPLIT_RE.sub(" ", title)
            for w in t_clean.split():
                w_norm = w.strip().lower()
                if len(w_norm) <= 2:
                    continue
                if w_norm in _STOPWORDS:
                    continue
                words.append(w_norm)

        if not words:
            continue

        counts = Counter(words)
        for word, count in counts.items():
            rows.append({"word": word, "decision": dec_str, "count": int(count)})

    if not rows:
        return pd.DataFrame(columns=["word", "decision", "count"])

    out = pd.DataFrame(rows)
    return out.sort_values("count", ascending=False, ignore_index=True)


# ============================================================================
# Utilidades para gráficos Altair
# ============================================================================


def decision_color(field: str = "decision") -> alt.Color:
    """
    Devuelve una escala de color fija por decisión para Altair.

    Nota:
    - Colores son constantes para mantener consistencia visual.
    """
    return alt.Color(
        f"{field}:N",
        title="Decisión",
        scale=alt.Scale(
            domain=["DELETE", "KEEP", "MAYBE", "UNKNOWN"],
            range=["#e53935", "#43a047", "#fbc02d", "#9e9e9e"],
        ),
    )


# ============================================================================
# Formateo de conteos + tamaño
# ============================================================================


def format_count_size(count: int, size_gb: float | int | None) -> str:
    """
    Devuelve un string tipo "N (X.XX GB)" si hay tamaño disponible.

    Si size_gb es None/NaN/no convertible, devuelve solo "N".
    """
    if size_gb is None or pd.isna(cast(Any, size_gb)):
        return str(int(count))

    try:
        size_f = float(size_gb)
    except Exception:
        return str(int(count))

    return f"{int(count)} ({size_f:.2f} GB)"