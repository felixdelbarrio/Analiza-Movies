from __future__ import annotations

"""
backend/decision_logic.py

Heurística para:
- Detectar posibles películas mal identificadas (misidentified).
- Ordenar filas filtradas (DELETE/MAYBE) para el CSV final.

Mejora aplicada (performance/robustez alineada con 2/3/4)
----------------------------------------------------------
Este módulo se invoca por película y suele ejecutarse miles de veces. La mejora aquí
no es “network-bound” como OMDb, pero sí puede acumular coste por:

- Normalización de strings repetida.
- difflib.SequenceMatcher (comparación relativamente cara) ejecutada en casos donde
  ya hay señales más fuertes.
- Ordenación de filas que pueden venir con tipos inconsistentes.

Por tanto:
1) Hacemos early-returns y short-circuit de cálculos caros.
2) Reducimos trabajo de difflib a los casos “ambiguos” donde puede aportar.
3) Robustecemos parseos y evitamos excepciones (no deben romper el pipeline).
4) Logging: mantenemos la filosofía del proyecto:
   - Este módulo no debe spamear.
   - Sólo debug si DEBUG_MODE=True y SILENT_MODE=False.
   - No hacemos cálculos extra para logs si no van a emitirse.

API pública
-----------
- detect_misidentified(...)-> str:
    Devuelve '' si no hay sospechas, o una cadena con hints separadas por " | ".
- sort_filtered_rows(rows)-> list[dict]:
    Ordena filas priorizando DELETE > MAYBE > KEEP > UNKNOWN y luego “impacto/certeza”.
"""

import difflib
import re
from collections.abc import Mapping
from typing import Final

from backend import logger as _logger
from backend.config_scoring import (
    IMDB_MIN_VOTES_FOR_KNOWN,
    IMDB_RATING_LOW_THRESHOLD,
    RT_RATING_LOW_THRESHOLD,
)

from backend.config_base import (
    DEBUG_MODE,
    SILENT_MODE,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Umbral de similitud (difflib ratio) por debajo del cual consideramos mismatch
TITLE_SIMILARITY_THRESHOLD: Final[float] = 0.60

# Regex precompilados (micro-optimización, evita recompilar por llamada)
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9\s]")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Límite razonable para comparar títulos (evita costes raros con strings enormes)
_MAX_TITLE_LEN_FOR_COMPARE: Final[int] = 180


# ============================================================
# Logging controlado por modos
# ============================================================

def _debug_enabled() -> bool:
    """True solo si el usuario quiere logs debug y NO estamos en silent."""
    return bool(DEBUG_MODE and not SILENT_MODE)


def _log_debug(msg: object) -> None:
    """
    Debug contextual:
    - DEBUG_MODE=False → no hace nada.
    - DEBUG_MODE=True:
        * SILENT_MODE=True: no emitimos (evitar ruido). Este módulo es “core heurístico”.
        * SILENT_MODE=False: usamos _logger.debug.
    """
    if not _debug_enabled():
        return
    try:
        _logger.debug(str(msg))
    except Exception:
        # Nunca romper por logging
        return


# ============================================================
# Helpers defensivos
# ============================================================

def _normalize_title(s: str | None) -> str:
    """
    Normaliza un título para comparación:
    - minúsculas
    - sin puntuación (solo [a-z0-9 ] tras normalizar)
    - espacios colapsados
    - truncado defensivo

    Nota:
    - Se usa para comparaciones:
        * contains
        * equality
        * difflib.SequenceMatcher
    """
    if not s:
        return ""

    # Truncado: protege de títulos patológicos o metadata corrupta
    s2 = s.strip()
    if len(s2) > _MAX_TITLE_LEN_FOR_COMPARE:
        s2 = s2[:_MAX_TITLE_LEN_FOR_COMPARE]

    s2 = s2.lower()
    s2 = _NON_ALNUM_RE.sub(" ", s2)
    s2 = _WS_RE.sub(" ", s2).strip()
    return s2


def _safe_imdb_id(value: object) -> str | None:
    """
    Normaliza un imdb id (tt1234567) a minúsculas, o None si inválido/vacío.
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

    if len(s) >= 4 and s[:4].isdigit():
        try:
            y = int(s[:4])
            if 1800 <= y <= 2200:
                return y
            return None
        except Exception:
            return None
    return None


def _clamp_int(v: object, default: int = 0) -> int:
    """
    Convierte a int seguro para ordenación.
    - bool -> int
    - float -> int (trunc)
    - str numérica -> int
    """
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
    except Exception:
        pass
    return default


def _clamp_float(v: object, default: float = 0.0) -> float:
    """
    Convierte a float seguro para ordenación.
    """
    try:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            return float(s)
    except Exception:
        pass
    return default


def _should_run_title_similarity(pt: str, ot: str) -> bool:
    """
    Decide si merece la pena ejecutar difflib.SequenceMatcher.

    difflib es relativamente caro, así que:
    - si ya son iguales / contains -> no hace falta.
    - si alguno está vacío -> no hace falta.
    - si son muy cortos -> ratio puede ser ruidoso; evitamos si < 4 chars.
    """
    if not pt or not ot:
        return False
    if pt == ot:
        return False
    if pt in ot or ot in pt:
        return False
    if len(pt) < 4 or len(ot) < 4:
        return False
    return True


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

    Importante:
    - Es una heurística: NO bloquea el pipeline.
    - Debe ser barata por llamada (se ejecuta miles de veces).

    Returns:
        '' si no hay sospechas, o hints separados por " | ".
    """
    if not omdb_data:
        return ""

    # Si OMDb no trae ficha válida, evitamos decisiones agresivas aquí.
    # Esto puede ocurrir si OMDb falló, o Response="False".
    if omdb_data.get("Response") != "True":
        return ""

    hints: list[str] = []

    plex_imdb = _safe_imdb_id(plex_imdb_id)
    omdb_imdb = _safe_imdb_id(omdb_data.get("imdbID"))

    # 0) Regla de oro (si coinciden, salimos rápido)
    if plex_imdb and omdb_imdb and plex_imdb == omdb_imdb:
        return ""

    # 1) IMDb mismatch (señal fuerte)
    if plex_imdb and omdb_imdb and plex_imdb != omdb_imdb:
        hints.append(f"IMDb mismatch: Plex={plex_imdb} vs OMDb={omdb_imdb}")

    # 2) Título OMDb y año OMDb
    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""
    omdb_year_int = _extract_omdb_year(omdb_data.get("Year"))

    pt_raw = plex_title if isinstance(plex_title, str) else ""
    pt = _normalize_title(pt_raw)
    ot = _normalize_title(omdb_title)

    # 3) Títulos claramente distintos (difflib sólo cuando aporte)
    if _should_run_title_similarity(pt, ot):
        # difflib.SequenceMatcher: coste moderado
        sim = difflib.SequenceMatcher(a=pt, b=ot).ratio()
        if _debug_enabled():
            _log_debug(f"Title similarity Plex vs OMDb: '{pt_raw}' vs '{omdb_title}' -> {sim:.2f}")

        if sim < TITLE_SIMILARITY_THRESHOLD:
            hints.append(f"Title mismatch: Plex='{pt_raw}' vs OMDb='{omdb_title}' (sim={sim:.2f})")

    # 4) Años muy diferentes (> 1)
    if plex_year is not None and omdb_year_int is not None:
        try:
            py = int(plex_year)
            if abs(py - omdb_year_int) > 1:
                hints.append(f"Year mismatch: Plex={py}, OMDb={omdb_year_int}")
        except Exception:
            if _debug_enabled():
                _log_debug(f"Could not compare years: plex_year={plex_year!r}, omdb_year={omdb_data.get('Year')!r}")

    # 5) Señales de “peli conocida” con rating muy bajo
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
      3) Mayor score bayesiano (si existe) o mayor rating IMDb.
      4) Mayor tamaño de fichero (más espacio a liberar primero).
      5) Título como último desempate (orden estable humano).

    Nota:
    - Normalmente se llama con filas DELETE/MAYBE, pero soporta cualquier decisión.
    - Debe ser determinista (importante para diffs y para que el usuario “no vea saltos”).
    """
    decision_rank_map: dict[str, int] = {"DELETE": 0, "MAYBE": 1, "KEEP": 2, "UNKNOWN": 3}

    def key_func(r: dict[str, object]) -> tuple[int, int, float, float, int, str]:
        decision_raw = r.get("decision")
        decision = decision_raw if isinstance(decision_raw, str) else "UNKNOWN"
        decision_rank = decision_rank_map.get(decision, 3)

        # Señales de certeza
        imdb_votes = _clamp_int(r.get("imdb_votes"), 0)

        # Preferimos bayes si existe (más robusto que rating a pelo)
        imdb_bayes = _clamp_float(r.get("imdb_bayes"), default=-1.0)
        imdb_rating = _clamp_float(r.get("imdb_rating"), 0.0)
        score_for_sort = imdb_bayes if imdb_bayes >= 0.0 else imdb_rating

        # Tamaño: en tu pipeline hay variantes:
        # - file_size (collection_analysis pone file_size=file_size_bytes)
        # - file_size_bytes
        file_size = _clamp_int(r.get("file_size"), 0)
        if file_size <= 0:
            file_size = _clamp_int(r.get("file_size_bytes"), 0)

        # Título como tiebreak (stable, humano)
        title_raw = r.get("title")
        title = title_raw if isinstance(title_raw, str) else ""

        # Orden:
        # - decision_rank asc
        # - votes desc
        # - score_for_sort desc
        # - imdb_rating desc (para casos sin bayes)
        # - file_size desc
        # - title asc
        return decision_rank, -imdb_votes, -score_for_sort, -imdb_rating, -file_size, title

    return sorted(rows, key=key_func)