from __future__ import annotations

"""
backend/decision_logic.py

Este módulo concentra reglas y heurísticas que afectan a:
- Detección de "misidentified" (posible película mal identificada).
- Ordenación determinista de filas filtradas para reporting (CSV/HTML).

⚠️ Importante (seguridad)
--------------------------
Aquí se generan señales que suelen acabar justificando decisiones de borrado aguas arriba.
Por eso el diseño prioriza:
- Reglas explicables (cada regla = id / nombre / severidad / mensaje).
- Determinismo (misma entrada => mismo output).
- Robustez (no lanzar, no romper pipeline).
- Logging centralizado y controlado (solo debug, best-effort).

Separación conceptual
---------------------
1) Reglas duras (hard rules):
   - Señales fuertes/deterministas de posible identificación errónea.
   - Ej: IMDb mismatch, año demasiado distinto, similitud de títulos muy baja.
2) Heurísticas suaves (soft rules):
   - Señales débiles/sospechas: pueden indicar "esto es una peli real pero mala"
     o simplemente un outlier.
   - Ej: rating muy bajo con muchos votos (película conocida y mal valorada).
   - Nota clave: esto NO implica misidentificación por sí mismo; solo sugiere revisar.

API pública (compat)
---------------------
- detect_misidentified(...)-> str
    Devuelve '' o un string con hints separados por " | ".
- sort_filtered_rows(rows)-> list[dict]
    Ordena filas priorizando DELETE > MAYBE > KEEP > UNKNOWN y luego “impacto/certeza”.

API extra (trazabilidad / testing)
-----------------------------------
- evaluate_misidentified_rules(...)-> list[RuleHit]
- summarize_rule_hits(hits)-> str

Configuración (backend.config_scoring)
--------------------------------------
Knobs usados aquí:
- DECISION_TITLE_SIMILARITY_THRESHOLD (float)
- DECISION_MAX_TITLE_LEN_FOR_COMPARE (int)
- DECISION_YEAR_MISMATCH_MAX_DELTA (int)
- DECISION_MIN_TITLE_LEN_FOR_DIFFLIB (int)              [nuevo]
- DECISION_OMDB_REQUIRE_RESPONSE_TRUE (bool)            [nuevo]

Optimización aplicada (6 puntos)
--------------------------------
1) OMDb usable configurable:
   - Response "true"/"TRUE" aceptado.
   - Si DECISION_OMDB_REQUIRE_RESPONSE_TRUE=False: fallback a imdbID/Title.
2) Regla de oro por IMDb:
   - Si Plex IMDb == OMDb IMDb => short-circuit (no hints).
   - Se deja debug interno si procede, sin contaminar salida.
3) Evitar difflib si ya hay señales hard fuertes:
   - Si IMDb mismatch o year mismatch, la similitud aporta poco y añade coste/ruido.
4) Parseo robusto del Year OMDb:
   - Extrae primer YYYY con regex de forma tolerante ("1994–1998", "1994 (USA)", etc.).
5) sort_filtered_rows: normaliza decision (strip/upper) para evitar rank incorrecto por casing.
6) sort_filtered_rows: helpers para minimizar trabajo dentro de key_func
   y mantener ordenación determinista y barata.
"""

import difflib
import re
from collections.abc import Mapping
from typing import Final, NamedTuple

from backend import logger as _logger
from backend.config_base import DEBUG_MODE, SILENT_MODE
from backend.config_scoring import (
    DECISION_MAX_TITLE_LEN_FOR_COMPARE,
    DECISION_MIN_TITLE_LEN_FOR_DIFFLIB,
    DECISION_OMDB_REQUIRE_RESPONSE_TRUE,
    DECISION_TITLE_SIMILARITY_THRESHOLD,
    DECISION_YEAR_MISMATCH_MAX_DELTA,
    IMDB_MIN_VOTES_FOR_KNOWN,
    IMDB_RATING_LOW_THRESHOLD,
    RT_RATING_LOW_THRESHOLD,
)

# -----------------------------------------------------------------------------
# Knobs (cacheados a nivel de módulo)
# -----------------------------------------------------------------------------
_TITLE_SIMILARITY_THRESHOLD: Final[float] = float(DECISION_TITLE_SIMILARITY_THRESHOLD)
_MAX_TITLE_LEN_FOR_COMPARE: Final[int] = int(DECISION_MAX_TITLE_LEN_FOR_COMPARE)
_YEAR_MISMATCH_MAX_DELTA: Final[int] = int(DECISION_YEAR_MISMATCH_MAX_DELTA)
_MIN_TITLE_LEN_FOR_DIFFLIB: Final[int] = int(DECISION_MIN_TITLE_LEN_FOR_DIFFLIB)
_OMDB_REQUIRE_RESPONSE_TRUE: Final[bool] = bool(DECISION_OMDB_REQUIRE_RESPONSE_TRUE)

# -----------------------------------------------------------------------------
# Regex precompilados (micro-optimización)
# -----------------------------------------------------------------------------
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9\s]")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_YEAR_4_RE: Final[re.Pattern[str]] = re.compile(r"(\d{4})")


# =============================================================================
# Logging centralizado
# =============================================================================

_LOG_TAG: Final[str] = "DECISION"


def _debug_enabled() -> bool:
    """
    Este módulo NO debe spamear:
    - Debug sólo si DEBUG_MODE=True y SILENT_MODE=False.
    """
    return bool(DEBUG_MODE and not SILENT_MODE)


def _dbg(msg: object) -> None:
    """
    Debug contextual best-effort usando logger central.
    Nunca rompe por logging.

    Nota:
    - No construyas strings caras fuera; pasa objetos simples.
    """
    if not _debug_enabled():
        return
    try:
        if hasattr(_logger, "debug_ctx"):
            _logger.debug_ctx(_LOG_TAG, msg)  # type: ignore[attr-defined]
        else:
            _logger.debug(str(msg))
    except Exception:
        return


# =============================================================================
# Tipos de reglas
# =============================================================================

class RuleHit(NamedTuple):
    """
    Resultado “explicable” de una regla.

    severity:
      - "hard": señal fuerte/determinista
      - "soft": señal heurística/sospecha
    """
    rule_id: str
    name: str
    severity: str  # "hard" | "soft"
    message: str


# =============================================================================
# Helpers defensivos
# =============================================================================

def _normalize_title(s: str | None) -> str:
    """
    Normaliza un título para comparación:
    - minúsculas
    - sin puntuación (solo [a-z0-9 ] tras normalizar)
    - espacios colapsados
    - truncado defensivo (configurable)

    Diseñado para:
    - comparaciones rápidas (==, contains)
    - y, si procede, difflib.SequenceMatcher
    """
    if not s:
        return ""
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
      - "1994-1998"
      - "1994 (USA)"

    Devuelve el primer año YYYY válido si existe.
    """
    if not isinstance(omdb_year_raw, str):
        return None
    s = omdb_year_raw.strip()
    if not s or s.upper() == "N/A":
        return None

    m = _YEAR_4_RE.search(s)
    if not m:
        return None
    try:
        y = int(m.group(1))
        return y if 1800 <= y <= 2200 else None
    except Exception:
        return None


def _should_run_title_similarity(pt: str, ot: str) -> bool:
    """
    Decide si merece la pena ejecutar difflib.SequenceMatcher.

    difflib es relativamente caro; lo evitamos si:
    - vacíos
    - iguales
    - contains (substrings)
    - demasiado cortos (configurable)
    """
    if not pt or not ot:
        return False
    if pt == ot:
        return False
    if pt in ot or ot in pt:
        return False
    if len(pt) < _MIN_TITLE_LEN_FOR_DIFFLIB or len(ot) < _MIN_TITLE_LEN_FOR_DIFFLIB:
        return False
    return True


def _is_omdb_usable(omdb_data: Mapping[str, object]) -> bool:
    """
    Determina si OMDb es "usable" para evaluar misidentified.

    Política:
    - Si DECISION_OMDB_REQUIRE_RESPONSE_TRUE=True (default conservador):
        usable si Response == "True" (case-insensitive)
    - Si False:
        usable si Response == True OR si hay imdbID/Title no vacío

    Objetivo:
    - Evitar falsos positivos con payloads tipo {"Response":"False","Error":"Movie not found!"}
    - Ser tolerante ante casing ("true"/"TRUE") y respuestas parciales si se permite.
    """
    resp = omdb_data.get("Response")
    if isinstance(resp, str) and resp.strip().lower() == "true":
        return True

    if _OMDB_REQUIRE_RESPONSE_TRUE:
        return False

    imdb_id = omdb_data.get("imdbID")
    if isinstance(imdb_id, str) and imdb_id.strip():
        return True

    title = omdb_data.get("Title")
    if isinstance(title, str) and title.strip():
        return True

    return False


# =============================================================================
# Reglas “misidentified” (explicables)
# =============================================================================

# --- Hard rules ids (señales fuertes)
RULE_IMDB_MATCH: Final[str] = "H001_IMDB_MATCH"
RULE_IMDB_MISMATCH: Final[str] = "H002_IMDB_MISMATCH"
RULE_YEAR_MISMATCH: Final[str] = "H003_YEAR_MISMATCH"
RULE_TITLE_MISMATCH: Final[str] = "H004_TITLE_MISMATCH"

# --- Soft rules ids (heurísticas)
RULE_LOW_IMDB_KNOWN: Final[str] = "S101_LOW_IMDB_KNOWN"
RULE_LOW_RT_KNOWN: Final[str] = "S102_LOW_RT_KNOWN"


def evaluate_misidentified_rules(
    *,
    plex_title: str | None,
    plex_year: int | None,
    plex_imdb_id: str | None,
    omdb_data: Mapping[str, object] | None,
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
) -> list[RuleHit]:
    """
    Evalúa reglas/hints de misidentificación y devuelve RuleHit[].

    Política:
    - Si OMDb no es usable => [].
    - Regla de oro: si IMDb coincide (Plex==OMDb) => [] (short-circuit),
      con debug interno opcional.

    Importante:
    - Las "soft rules" NO significan misidentificación; solo alertan de casos que
      podrían merecer revisión.
    """
    if not omdb_data:
        return []

    if not _is_omdb_usable(omdb_data):
        return []

    plex_imdb = _safe_imdb_id(plex_imdb_id)
    omdb_imdb = _safe_imdb_id(omdb_data.get("imdbID"))

    # Regla de oro: si coincide, NO hay misidentified
    if plex_imdb and omdb_imdb and plex_imdb == omdb_imdb:
        _dbg(f"misidentified shortcircuit | {RULE_IMDB_MATCH} imdb={plex_imdb}")
        return []

    hits: list[RuleHit] = []

    # Hard: IMDb mismatch
    hard_imdb_mismatch = False
    if plex_imdb and omdb_imdb and plex_imdb != omdb_imdb:
        hard_imdb_mismatch = True
        hits.append(
            RuleHit(
                RULE_IMDB_MISMATCH,
                "IMDb mismatch",
                "hard",
                f"IMDb mismatch: Plex={plex_imdb} vs OMDb={omdb_imdb}",
            )
        )

    # Preparar título/año OMDb
    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""
    omdb_year_int = _extract_omdb_year(omdb_data.get("Year"))

    pt_raw = plex_title if isinstance(plex_title, str) else ""
    pt = _normalize_title(pt_raw)
    ot = _normalize_title(omdb_title)

    # Hard: año muy diferente
    hard_year_mismatch = False
    if plex_year is not None and omdb_year_int is not None:
        try:
            py = int(plex_year)
            delta = abs(py - int(omdb_year_int))
            if delta > _YEAR_MISMATCH_MAX_DELTA:
                hard_year_mismatch = True
                hits.append(
                    RuleHit(
                        RULE_YEAR_MISMATCH,
                        "Year mismatch",
                        "hard",
                        f"Year mismatch: Plex={py}, OMDb={omdb_year_int} (delta={delta})",
                    )
                )
        except Exception:
            _dbg(f"year compare failed | plex_year={plex_year!r} omdb_year={omdb_data.get('Year')!r}")

    # Evitar difflib si ya hay señales hard fuertes (optimización + menos ruido)
    if not (hard_imdb_mismatch or hard_year_mismatch):
        if _should_run_title_similarity(pt, ot):
            sim = difflib.SequenceMatcher(a=pt, b=ot).ratio()
            _dbg(f"title similarity | plex='{pt_raw}' omdb='{omdb_title}' sim={sim:.2f}")

            if sim < _TITLE_SIMILARITY_THRESHOLD:
                hits.append(
                    RuleHit(
                        RULE_TITLE_MISMATCH,
                        "Title mismatch",
                        "hard",
                        f"Title mismatch: Plex='{pt_raw}' vs OMDb='{omdb_title}' (sim={sim:.2f})",
                    )
                )

    # Soft: “peli conocida” con rating muy bajo
    votes = int(imdb_votes) if isinstance(imdb_votes, int) else 0
    if (
        imdb_rating is not None
        and float(imdb_rating) <= float(IMDB_RATING_LOW_THRESHOLD)
        and votes >= int(IMDB_MIN_VOTES_FOR_KNOWN)
    ):
        hits.append(
            RuleHit(
                RULE_LOW_IMDB_KNOWN,
                "Low IMDb for known movie",
                "soft",
                (
                    f"IMDb muy baja ({float(imdb_rating):.1f} ≤ {IMDB_RATING_LOW_THRESHOLD}) "
                    f"con muchos votos ({votes} ≥ {IMDB_MIN_VOTES_FOR_KNOWN}). Revisar identificación."
                ),
            )
        )

    # Soft: RT muy bajo con suficientes votos
    if (
        rt_score is not None
        and int(rt_score) <= int(RT_RATING_LOW_THRESHOLD)
        and votes >= int(IMDB_MIN_VOTES_FOR_KNOWN)
    ):
        hits.append(
            RuleHit(
                RULE_LOW_RT_KNOWN,
                "Low RT for known movie",
                "soft",
                (
                    f"RT muy bajo ({int(rt_score)}% ≤ {RT_RATING_LOW_THRESHOLD}%) "
                    f"para una peli aparentemente conocida ({votes} votos IMDb)."
                ),
            )
        )

    return hits


def summarize_rule_hits(hits: list[RuleHit]) -> str:
    """
    Convierte RuleHit[] a string estable para UI/logs.

    Política:
    - Ordena por severidad (hard primero), luego por rule_id para determinismo.
    - Une mensajes con " | " (compat con API antigua).
    """
    if not hits:
        return ""
    sev_rank = {"hard": 0, "soft": 1}
    hits_sorted = sorted(hits, key=lambda h: (sev_rank.get(h.severity, 9), h.rule_id))
    return " | ".join(h.message for h in hits_sorted if h.message)


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
    Compat wrapper: devuelve '' o string con hints separados por " | ".

    Nota:
    - Este módulo NO decide DELETE por sí mismo.
      Devuelve señales explicables que capas superiores pueden usar para justificar acciones.
    """
    hits = evaluate_misidentified_rules(
        plex_title=plex_title,
        plex_year=plex_year,
        plex_imdb_id=plex_imdb_id,
        omdb_data=omdb_data,
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
    )
    return summarize_rule_hits(hits)


# =============================================================================
# Ordenación determinista (reporting)
# =============================================================================

def _clamp_int(v: object, default: int = 0) -> int:
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


def _norm_decision(v: object) -> str:
    """
    Normaliza decision a mayúsculas/strip para evitar ranks incorrectos.
    """
    if not isinstance(v, str):
        return "UNKNOWN"
    s = v.strip().upper()
    return s if s else "UNKNOWN"


def _get_file_size(r: Mapping[str, object]) -> int:
    """
    Obtiene tamaño de fichero con fallback (compat pipeline):
    - file_size
    - file_size_bytes
    """
    fs = _clamp_int(r.get("file_size"), 0)
    if fs > 0:
        return fs
    return _clamp_int(r.get("file_size_bytes"), 0)


def sort_filtered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Ordena filas para el CSV filtrado, priorizando:
      1) decision: DELETE, MAYBE, KEEP, UNKNOWN
      2) imdb_votes desc
      3) score bayes desc (si existe) o imdb_rating desc
      4) file_size desc
      5) title asc
    """
    decision_rank_map: dict[str, int] = {"DELETE": 0, "MAYBE": 1, "KEEP": 2, "UNKNOWN": 3}

    def key_func(r: dict[str, object]) -> tuple[int, int, float, float, int, str]:
        decision = _norm_decision(r.get("decision"))
        decision_rank = decision_rank_map.get(decision, 3)

        votes = _clamp_int(r.get("imdb_votes"), 0)

        bayes = _clamp_float(r.get("imdb_bayes"), default=-1.0)
        rating = _clamp_float(r.get("imdb_rating"), 0.0)
        score_for_sort = bayes if bayes >= 0.0 else rating

        file_size = _get_file_size(r)

        title_raw = r.get("title")
        title = title_raw if isinstance(title_raw, str) else ""

        return decision_rank, -votes, -score_for_sort, -rating, -file_size, title

    return sorted(rows, key=key_func)


__all__ = [
    "detect_misidentified",
    "sort_filtered_rows",
    "evaluate_misidentified_rules",
    "summarize_rule_hits",
    "RuleHit",
]