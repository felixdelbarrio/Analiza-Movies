"""
backend/decision_logic.py

Heurística para:
- Detectar posibles películas mal identificadas (misidentified).
- Ordenar filas filtradas (DELETE/MAYBE) para el CSV final.

Filosofía (alineada con el logger del proyecto)
-----------------------------------------------
- Este módulo NO debería “ensuciar” salida en ejecuciones normales.
- En DEBUG_MODE=True podemos emitir trazas, pero:
  - SILENT_MODE=True: nada de spam; si hace falta, que lo hagan orquestadores con progress.
  - SILENT_MODE=False: debug/info permitido.
- Por eso aquí usamos un helper `_log_debug(...)` que respeta DEBUG_MODE y SILENT_MODE,
  y además evita cálculos caros si no van a mostrarse.

API pública
-----------
- detect_misidentified(...)-> str:
    Devuelve '' si no hay sospechas, o una cadena con hints separadas por " | ".
- sort_filtered_rows(rows)-> list[dict]:
    Ordena filas priorizando DELETE > MAYBE > KEEP > UNKNOWN y luego “impacto/certeza”.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from typing import Final

from backend import logger as _logger
from backend.config import (
    DEBUG_MODE,
    IMDB_MIN_VOTES_FOR_KNOWN,
    IMDB_RATING_LOW_THRESHOLD,
    RT_RATING_LOW_THRESHOLD,
    SILENT_MODE,
)

TITLE_SIMILARITY_THRESHOLD: Final[float] = 0.60

# Regex precompilados (micro-optimización, evita recompilar por llamada)
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9\s]")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


# ============================================================
# Logging controlado por modos
# ============================================================

def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: no emitimos (evitar ruido). Este módulo es “core heurístico”.
        * SILENT_MODE=False: usamos _logger.debug.
    """
    if not DEBUG_MODE:
        return
    if SILENT_MODE:
        return
    try:
        _logger.debug(str(msg))
    except Exception:
        pass


# ============================================================
# Helpers defensivos
# ============================================================

def _normalize_title(s: str | None) -> str:
    """
    Normaliza un título para comparación:
    - minúsculas
    - sin puntuación
    - espacios colapsados

    Nota:
    - Se usa para difflib.SequenceMatcher y comparaciones tipo contains.
    - Si el título está vacío -> "".
    """
    if not s:
        return ""
    s2 = s.lower()
    s2 = _NON_ALNUM_RE.sub(" ", s2)
    s2 = _WS_RE.sub(" ", s2).strip()
    return s2


def _safe_imdb_id(value: object) -> str | None:
    """
    Normaliza un imdb id (tt1234567) a minúsculas, o None si inválido.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _extract_omdb_year(omdb_year_raw: object) -> int | None:
    """
    OMDb Year puede venir como:
      - "1994"
      - "1994–1998"
      - "N/A"
    Devolvemos el primer año (4 dígitos) si existe.
    """
    if not isinstance(omdb_year_raw, str):
        return None
    s = omdb_year_raw.strip()
    if not s or s.upper() == "N/A":
        return None
    # coger primeros 4 si son dígitos
    if len(s) >= 4 and s[:4].isdigit():
        try:
            return int(s[:4])
        except Exception:
            return None
    return None


def _clamp_int(v: object, default: int = 0) -> int:
    """
    Convierte a int seguro para ordenación.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return default


def _clamp_float(v: object, default: float = 0.0) -> float:
    """
    Convierte a float seguro para ordenación.
    """
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    return default


# ============================================================
# API pública: misidentified
# ============================================================

def detect_misidentified(
    plex_title: str | None,
    plex_year: int | None,
    plex_imdb_id: str | None,
    omdb_data: Mapping[str, object] | None,
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
) -> str:
    """
    Detecta posibles identificaciones erróneas comparando:
    - imdbID Plex vs imdbID OMDb (regla de oro: si coinciden → NO misidentified)
    - título Plex vs título OMDb (similitud baja)
    - año Plex vs año OMDb (diferencia > 1)
    - rating muy bajo con “suficientes votos” (peli conocida con nota sospechosa)
    - RT muy bajo con “suficientes votos”

    Parámetros:
        plex_title:
            Título “display” del origen (Plex). Puede diferir del buscado.
        plex_year:
            Año del origen (Plex).
        plex_imdb_id:
            imdbID detectado por Plex (si existe).
        omdb_data:
            Payload OMDb (Mapping). Si Response != "True" → devolvemos "".
        imdb_rating/imdb_votes/rt_score:
            Señales ya normalizadas (idealmente extraídas por extract_ratings_from_omdb).

    Returns:
        str:
            '' si no hay sospechas,
            o texto con hints separadas por " | ".
    """
    if not omdb_data:
        return ""

    # Si OMDb no trae ficha válida, no forzamos misidentified aquí.
    if omdb_data.get("Response") != "True":
        return ""

    hints: list[str] = []

    plex_imdb = _safe_imdb_id(plex_imdb_id)
    omdb_imdb = _safe_imdb_id(omdb_data.get("imdbID"))

    # 0) Regla de oro
    if plex_imdb and omdb_imdb and plex_imdb == omdb_imdb:
        return ""

    # 1) IMDb mismatch (señal fuerte)
    if plex_imdb and omdb_imdb and plex_imdb != omdb_imdb:
        hints.append(f"IMDb mismatch: Plex={plex_imdb} vs OMDb={omdb_imdb}")

    # 2) Datos OMDb básicos
    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""
    omdb_year_int = _extract_omdb_year(omdb_data.get("Year"))

    pt = _normalize_title(plex_title)
    ot = _normalize_title(omdb_title)

    # 3) Títulos claramente distintos (solo si hay señal suficiente)
    if pt and ot and pt != ot and pt not in ot and ot not in pt:
        # difflib puede ser relativamente caro; lo hacemos solo si vamos a usarlo
        sim = difflib.SequenceMatcher(a=pt, b=ot).ratio()
        _log_debug(f"Title similarity Plex vs OMDb: '{plex_title}' vs '{omdb_title}' -> {sim:.2f}")
        if sim < TITLE_SIMILARITY_THRESHOLD:
            hints.append(
                f"Title mismatch: Plex='{plex_title}' vs OMDb='{omdb_title}' (sim={sim:.2f})"
            )

    # 4) Años muy diferentes (> 1)
    try:
        if plex_year is not None and omdb_year_int is not None:
            plex_year_int = int(plex_year)
            if abs(plex_year_int - omdb_year_int) > 1:
                hints.append(f"Year mismatch: Plex={plex_year_int}, OMDb={omdb_year_int}")
    except Exception:
        _log_debug(
            f"Could not compare years: plex_year={plex_year!r}, omdb_year={omdb_data.get('Year')!r}"
        )

    # 5) IMDb muy baja con suficientes votos (posible “otra” peli)
    votes: int = imdb_votes if isinstance(imdb_votes, int) else 0
    if (
        imdb_rating is not None
        and imdb_rating <= IMDB_RATING_LOW_THRESHOLD
        and votes >= IMDB_MIN_VOTES_FOR_KNOWN
    ):
        hints.append(
            (
                f"IMDb muy baja ({imdb_rating:.1f} ≤ {IMDB_RATING_LOW_THRESHOLD}) "
                f"con bastantes votos ({votes} ≥ {IMDB_MIN_VOTES_FOR_KNOWN}). "
                "Revisar identificación."
            )
        )

    # 6) RT muy bajo con suficientes votos (usa el mismo umbral de “conocida”)
    if (
        rt_score is not None
        and rt_score <= RT_RATING_LOW_THRESHOLD
        and votes >= IMDB_MIN_VOTES_FOR_KNOWN
    ):
        hints.append(
            (
                f"RT muy bajo ({rt_score}% ≤ {RT_RATING_LOW_THRESHOLD}%) "
                f"para una peli aparentemente conocida ({votes} votos IMDb)."
            )
        )

    return " | ".join(hints)


# ============================================================
# API pública: ordenación
# ============================================================

def sort_filtered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Ordena filas para el CSV filtrado, priorizando:

      1) decision: DELETE primero, luego MAYBE, luego KEEP, luego UNKNOWN.
      2) Más votos IMDb (más “certeza” / relevancia).
      3) Mayor rating IMDb.
      4) Mayor tamaño de fichero (más espacio a liberar primero).

    Nota:
    - Esta función se usa sobre todo con filas ya filtradas (DELETE/MAYBE),
      pero soporta cualquier decisión.
    """

    decision_rank_map: dict[str, int] = {"DELETE": 0, "MAYBE": 1, "KEEP": 2, "UNKNOWN": 3}

    def key_func(r: dict[str, object]) -> tuple[int, int, float, int]:
        decision_raw = r.get("decision")
        decision = decision_raw if isinstance(decision_raw, str) else "UNKNOWN"
        decision_rank = decision_rank_map.get(decision, 3)

        # En tus rows hay variantes históricas:
        # - imdb_votes (core) suele ser int o None
        # - file_size puede ser "file_size" o "file_size_bytes" según orquestador
        imdb_votes = _clamp_int(r.get("imdb_votes"), 0)
        imdb_rating = _clamp_float(r.get("imdb_rating"), 0.0)

        file_size = _clamp_int(r.get("file_size"), 0)
        if file_size <= 0:
            file_size = _clamp_int(r.get("file_size_bytes"), 0)

        # Nota: usamos negativos para ordenar desc.
        return decision_rank, -imdb_votes, -imdb_rating, -file_size

    return sorted(rows, key=key_func)