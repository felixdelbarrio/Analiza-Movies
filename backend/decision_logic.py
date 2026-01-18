"""
backend/decision_logic.py

Actualización (multi-idioma):
- Usa backend.title_utils para normalización de títulos.
- Evita comparar título Plex localizado (ES/FR/IT) vs OMDb (a menudo EN) -> reduce falsos positivos.
"""

from __future__ import annotations

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
from backend.title_utils import (
    NormalizeOptions,
    normalize_title_for_compare,
    should_skip_title_similarity_due_to_language,
)

# -----------------------------------------------------------------------------
# Knobs (cacheados a nivel de módulo)
# -----------------------------------------------------------------------------
_TITLE_SIMILARITY_THRESHOLD: Final[float] = float(DECISION_TITLE_SIMILARITY_THRESHOLD)
_MAX_TITLE_LEN_FOR_COMPARE: Final[int] = int(DECISION_MAX_TITLE_LEN_FOR_COMPARE)
_YEAR_MISMATCH_MAX_DELTA: Final[int] = int(DECISION_YEAR_MISMATCH_MAX_DELTA)
_MIN_TITLE_LEN_FOR_DIFFLIB: Final[int] = int(DECISION_MIN_TITLE_LEN_FOR_DIFFLIB)
_OMDB_REQUIRE_RESPONSE_TRUE: Final[bool] = bool(DECISION_OMDB_REQUIRE_RESPONSE_TRUE)

_YEAR_4_RE: Final[re.Pattern[str]] = re.compile(r"(\d{4})")


# =============================================================================
# Logging centralizado
# =============================================================================

_LOG_TAG: Final[str] = "DECISION"


def _debug_enabled() -> bool:
    return bool(DEBUG_MODE and not SILENT_MODE)


def _dbg(msg: object) -> None:
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
    rule_id: str
    name: str
    severity: str  # "hard" | "soft"
    message: str


# =============================================================================
# Helpers defensivos
# =============================================================================


def _safe_imdb_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _extract_omdb_year(omdb_year_raw: object) -> int | None:
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


def _should_run_title_similarity(pt_norm: str, ot_norm: str) -> bool:
    if not pt_norm or not ot_norm:
        return False
    if pt_norm == ot_norm:
        return False
    if pt_norm in ot_norm or ot_norm in pt_norm:
        return False
    if (
        len(pt_norm) < _MIN_TITLE_LEN_FOR_DIFFLIB
        or len(ot_norm) < _MIN_TITLE_LEN_FOR_DIFFLIB
    ):
        return False
    return True


def _is_omdb_usable(omdb_data: Mapping[str, object]) -> bool:
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


def _title_similarity(pt_norm: str, ot_norm: str) -> tuple[float | None, bool]:
    """
    Devuelve (similarity, compared):
      - compared=False => se decidió NO comparar (títulos cortos / substring / vacíos / iguales)
      - compared=True  => se intentó comparar; similarity puede ser None si falló.
    """
    if not _should_run_title_similarity(pt_norm, ot_norm):
        return None, False
    try:
        return difflib.SequenceMatcher(a=pt_norm, b=ot_norm).ratio(), True
    except Exception:
        return None, True


def _votes_as_int(v: object) -> int:
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
    except Exception:
        pass
    return 0


# =============================================================================
# Reglas “misidentified” (explicables)
# =============================================================================

RULE_IMDB_MATCH: Final[str] = "H001_IMDB_MATCH"
RULE_IMDB_MISMATCH: Final[str] = "H002_IMDB_MISMATCH"
RULE_YEAR_MISMATCH: Final[str] = "H003_YEAR_MISMATCH"
RULE_TITLE_MISMATCH: Final[str] = "H004_TITLE_MISMATCH"

RULE_IMDB_MATCH_METADATA_DIVERGE: Final[str] = "S001_IMDB_MATCH_METADATA_DIVERGE"
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
    if not omdb_data:
        return []
    if not _is_omdb_usable(omdb_data):
        return []

    plex_imdb = _safe_imdb_id(plex_imdb_id)
    omdb_imdb = _safe_imdb_id(omdb_data.get("imdbID"))

    omdb_title_raw = omdb_data.get("Title")
    omdb_title = omdb_title_raw if isinstance(omdb_title_raw, str) else ""
    omdb_year_int = _extract_omdb_year(omdb_data.get("Year"))

    pt_raw = plex_title if isinstance(plex_title, str) else ""

    # Normalización compare centralizada (recorte defensivo)
    norm_opts = NormalizeOptions(
        max_len=_MAX_TITLE_LEN_FOR_COMPARE, strip_accents=False
    )
    pt_norm = normalize_title_for_compare(pt_raw, options=norm_opts)
    ot_norm = normalize_title_for_compare(omdb_title, options=norm_opts)

    # ------------------------------------------------------------
    # Regla de oro: IMDb match => NO hard rules, pero sí soft guard
    # (con guard multi-idioma para no comparar ES vs EN)
    # ------------------------------------------------------------
    if plex_imdb and omdb_imdb and plex_imdb == omdb_imdb:
        _dbg(f"misidentified imdb-match | {RULE_IMDB_MATCH} imdb={plex_imdb}")

        year_delta: int | None = None
        if plex_year is not None and omdb_year_int is not None:
            try:
                year_delta = abs(int(plex_year) - int(omdb_year_int))
            except Exception:
                year_delta = None

        # Guard idioma: si OMDb parece EN y Plex no parece EN, no comparo por título.
        skip_title = should_skip_title_similarity_due_to_language(pt_raw, omdb_title)

        sim: float | None = None
        compared = False
        title_bad = False
        cheap_title_diverge = False

        if not skip_title:
            sim, compared = _title_similarity(pt_norm, ot_norm)
            title_bad = sim is not None and sim < _TITLE_SIMILARITY_THRESHOLD

            # ✅ Importante: "diverge" solo si REALMENTE hemos intentado comparar
            # (evita falsos positivos cuando la comparación se "skipped" por títulos cortos/substrings).
            if compared and sim is None and pt_norm and ot_norm and pt_norm != ot_norm:
                cheap_title_diverge = True

        year_bad = year_delta is not None and year_delta > _YEAR_MISMATCH_MAX_DELTA

        if title_bad or year_bad or cheap_title_diverge:
            parts: list[str] = []
            if year_bad:
                parts.append(
                    f"Year Plex={plex_year} vs OMDb={omdb_year_int} (delta={year_delta})"
                )
            if not skip_title:
                if title_bad and sim is not None:
                    parts.append(
                        f"Title sim={sim:.2f} Plex='{pt_raw}' vs OMDb='{omdb_title}'"
                    )
                elif cheap_title_diverge:
                    parts.append(
                        f"Title compare failed Plex='{pt_raw}' vs OMDb='{omdb_title}'"
                    )
            else:
                parts.append("Title compare skipped (likely localized vs EN)")

            msg = "IMDb match but metadata diverges: " + "; ".join(parts)
            _dbg(f"soft guard hit | {RULE_IMDB_MATCH_METADATA_DIVERGE} | {msg}")
            return [
                RuleHit(
                    RULE_IMDB_MATCH_METADATA_DIVERGE,
                    "IMDb match but metadata diverges",
                    "soft",
                    msg,
                )
            ]

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
            _dbg(
                f"year compare failed | plex_year={plex_year!r} omdb_year={omdb_data.get('Year')!r}"
            )

    # Hard: título mismatch (solo si no hay señales hard fuertes, y con guard multi-idioma)
    if not (hard_imdb_mismatch or hard_year_mismatch):
        skip_title = should_skip_title_similarity_due_to_language(pt_raw, omdb_title)
        if not skip_title:
            sim, compared = _title_similarity(pt_norm, ot_norm)
            if compared and sim is not None:
                _dbg(
                    f"title similarity | plex='{pt_raw}' omdb='{omdb_title}' sim={sim:.2f}"
                )
                if sim < _TITLE_SIMILARITY_THRESHOLD:
                    hits.append(
                        RuleHit(
                            RULE_TITLE_MISMATCH,
                            "Title mismatch",
                            "hard",
                            f"Title mismatch: Plex='{pt_raw}' vs OMDb='{omdb_title}' (sim={sim:.2f})",
                        )
                    )
        else:
            _dbg("title compare skipped (likely localized vs EN)")

    # Soft: “peli conocida” con rating muy bajo
    votes = _votes_as_int(imdb_votes) if imdb_votes is not None else 0
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
    if not isinstance(v, str):
        return "UNKNOWN"
    s = v.strip().upper()
    return s if s else "UNKNOWN"


def _get_file_size(r: Mapping[str, object]) -> int:
    fs = _clamp_int(r.get("file_size"), 0)
    if fs > 0:
        return fs
    return _clamp_int(r.get("file_size_bytes"), 0)


def sort_filtered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    decision_rank_map: dict[str, int] = {
        "DELETE": 0,
        "MAYBE": 1,
        "KEEP": 2,
        "UNKNOWN": 3,
    }

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
