from __future__ import annotations

from collections import Counter
import json
import re
from typing import Iterable, List

import altair as alt
import pandas as pd


# -------------------------------------------------------------------
# Utilidades de JSON
# -------------------------------------------------------------------


def safe_json_loads_single(x: object) -> dict | list | None:
    """Parsea JSON de forma segura para una sola celda/campo.

    - Si es str no vacía → intenta json.loads, devuelve dict/list o None si falla.
    - Si ya es dict/list → lo devuelve tal cual.
    - En cualquier otro caso → None.
    """
    if isinstance(x, (dict, list)):
        return x

    if isinstance(x, str) and x.strip():
        try:
            return json.loads(x)
        except Exception:
            return None

    return None


# -------------------------------------------------------------------
# Columnas derivadas para el dashboard
# -------------------------------------------------------------------


_NUMERIC_COLS: tuple[str, ...] = (
    "imdb_rating",
    "rt_score",
    "imdb_votes",
    "year",
    "plex_rating",
    "file_size",
)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas derivadas ligeras: tamaños, década, etc.

    No modifica el DataFrame original: siempre devuelve una copia.
    """
    df = df.copy()

    # Aseguramos tipos numéricos en columnas básicas (si existen)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tamaño en GB (si hay file_size en bytes)
    if "file_size" in df.columns:
        # Usamos to_numeric una vez más por si en el futuro llega como str
        file_size_num = pd.to_numeric(df["file_size"], errors="coerce")
        df["file_size_gb"] = file_size_num.astype("float64") / (1024**3)

    # Década y etiqueta de década
    if "year" in df.columns:
        year_num = pd.to_numeric(df["year"], errors="coerce")
        decade = (year_num // 10) * 10  # manteniendo índices

        df["decade"] = decade

        def _format_decade(val: float | int | None) -> str | None:
            if pd.isna(val):
                return None
            try:
                return f"{int(val)}s"
            except Exception:
                return None

        df["decade_label"] = df["decade"].apply(_format_decade)

    return df


# -------------------------------------------------------------------
# Explosión de géneros desde omdb_json
# -------------------------------------------------------------------


def explode_genres_from_omdb_json(df: pd.DataFrame) -> pd.DataFrame:
    """Construye un DataFrame 'exploded' por género usando la columna omdb_json.

    Si no existe la columna omdb_json, devuelve un DataFrame vacío con
    las mismas columnas + 'genre'.
    """
    if "omdb_json" not in df.columns:
        return pd.DataFrame(columns=[*df.columns, "genre"])

    df_g = df.copy().reset_index(drop=True)

    def extract_genre(raw: object) -> list[str]:
        data = safe_json_loads_single(raw)
        if not isinstance(data, dict):
            return []
        g = data.get("Genre")
        if not g:
            return []
        return [x.strip() for x in str(g).split(",") if x.strip()]

    df_g["genre_list"] = df_g["omdb_json"].apply(extract_genre)
    df_g = df_g.explode("genre_list").reset_index(drop=True)
    df_g = df_g.rename(columns={"genre_list": "genre"})

    mask = df_g["genre"].notna() & (df_g["genre"] != "")
    return df_g.loc[mask].copy()


# -------------------------------------------------------------------
# Recuento de palabras en títulos por decisión
# -------------------------------------------------------------------


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


def build_word_counts(df: pd.DataFrame, decisions: List[str] | Iterable[str]) -> pd.DataFrame:
    """Construye un DataFrame con recuento de palabras en títulos.

    - Filtra por las decisiones indicadas (DELETE/MAYBE, etc.).
    - Elimina stopwords básicas y palabras de longitud <= 2.
    - Devuelve columnas: word, decision, count
    """
    if "decision" not in df.columns or "title" not in df.columns:
        return pd.DataFrame(columns=["word", "decision", "count"])

    decisions_set = set(decisions)
    df = df[df["decision"].isin(decisions_set)].copy()
    if df.empty:
        return pd.DataFrame(columns=["word", "decision", "count"])

    rows: list[dict[str, object]] = []

    for dec, sub in df.groupby("decision"):
        words: list[str] = []
        for title in sub["title"].dropna().astype(str):
            # Quitamos puntuación y separamos en tokens
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
            rows.append({"word": word, "decision": dec, "count": count})

    if not rows:
        return pd.DataFrame(columns=["word", "decision", "count"])

    out = pd.DataFrame(rows)
    return out.sort_values("count", ascending=False, ignore_index=True)


# -------------------------------------------------------------------
# Utilidades para gráficos Altair
# -------------------------------------------------------------------


def decision_color(field: str = "decision") -> alt.Color:
    """Devuelve una escala de color fija por decisión para Altair."""
    return alt.Color(
        f"{field}:N",
        title="Decisión",
        scale=alt.Scale(
            domain=["DELETE", "KEEP", "MAYBE", "UNKNOWN"],
            range=["#e53935", "#43a047", "#fbc02d", "#9e9e9e"],
        ),
    )


# -------------------------------------------------------------------
# Formateo de conteos + tamaño
# -------------------------------------------------------------------


def format_count_size(count: int, size_gb: float | int | None) -> str:
    """Devuelve un string tipo 'N (X.XX GB)' si hay tamaño disponible."""
    if size_gb is None or pd.isna(size_gb):
        return str(count)
    try:
        size_f = float(size_gb)
    except Exception:
        return str(count)
    return f"{count} ({size_f:.2f} GB)"