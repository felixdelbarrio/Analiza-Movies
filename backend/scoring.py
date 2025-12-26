from __future__ import annotations

"""
backend/scoring.py

Motor de decisión (KEEP / DELETE / MAYBE / UNKNOWN) basado en señales de calidad:
- IMDb (rating + votos) como señal principal, vía score bayesiano conservador.
- Rotten Tomatoes (RT) como señal secundaria (boost / confirmación / desempate).
- Metacritic como señal de crítica: **solo refuerza la explicación**, nunca cambia
  la decisión por sí sola (regla conservadora).

Objetivos (diseño)
------------------
1) Estabilidad:
   - Evitar “flip-flop” cerca de umbrales (margen en desempates RT).
   - Evitar falsos positivos de DELETE con pocos votos.

2) Explicabilidad:
   - `reason`: texto humano
   - `rule`: etiqueta estable para auditoría/tests
   - `inputs`: snapshot de señales/umbrales usados

3) Robustez / degradación:
   - Si faltan señales, degradar a MAYBE/UNKNOWN sin romper el pipeline.
   - `compute_scoring()` nunca debe lanzar excepción.

Compatibilidad con Lazy OMDb (mejora de performance)
----------------------------------------------------
Con la estrategia “lazy OMDb”, muchas llamadas al core pasarán:
- imdb_rating=None, imdb_votes=None, rt_score=None, metacritic_score=None

Este módulo debe:
- devolver rápidamente UNKNOWN/MAYBE con explicación,
- sin depender de OMDb.
"""

from typing import Final, TypeAlias

from backend.config_scoring import (
    BAYES_DELETE_MAX_SCORE,
    IMDB_DELETE_MAX_RATING,
    IMDB_KEEP_MIN_RATING,
    IMDB_KEEP_MIN_RATING_WITH_RT,
    IMDB_MIN_VOTES_FOR_KNOWN,
    METACRITIC_DELETE_MAX_SCORE,
    METACRITIC_KEEP_MIN_SCORE,
    RT_DELETE_MAX_SCORE,
    RT_KEEP_MIN_SCORE,
    get_votes_threshold_for_year,
)
from backend.stats import (
    get_auto_delete_rating_threshold,
    get_auto_keep_rating_threshold,
    get_global_imdb_mean_from_cache,
)

ScoringDict: TypeAlias = dict[str, object]

# Pequeño margen para desempates cerca del umbral de DELETE (evita flip-flop).
_RT_TIEBREAK_BAYES_MARGIN: Final[float] = 0.30

# Decisiones válidas esperadas por el pipeline
_VALID_DECISIONS: Final[set[str]] = {"KEEP", "MAYBE", "DELETE", "UNKNOWN"}


# ============================================================================
# Helpers defensivos de parsing
# ============================================================================

def _safe_int(value: object) -> int | None:
    """
    Convierte a int de forma defensiva.
    Soporta:
      - int, float
      - strings tipo "12,345"
    """
    try:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if not s or s.upper() == "N/A":
                return None
            return int(s)
        return None
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    """
    Convierte a float de forma defensiva.
    Soporta:
      - int, float
      - strings tipo "7.3"
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s or s.upper() == "N/A":
                return None
            return float(s)
        return None
    except Exception:
        return None


def _clamp_float(value: float, *, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _decision_or_unknown(value: object) -> str:
    """
    Normaliza decision a un set finito esperado por el pipeline.
    """
    if value is None:
        return "UNKNOWN"
    d = str(value).strip().upper()
    return d if d in _VALID_DECISIONS else "UNKNOWN"


# ============================================================================
# Score bayesiano
# ============================================================================

def _compute_bayes_score(
    imdb_rating: float | None,
    imdb_votes: int | None,
    *,
    m: int,
    c_global: float,
) -> float | None:
    """
    Calcula el score bayesiano IMDb:

        score_bayes = (v / (v + m)) * R + (m / (v + m)) * C

    Donde:
      - R: imdb_rating (rating observado)
      - v: imdb_votes (votos observados)
      - m: fuerza del prior (votos esperados / mínimo por antigüedad)
      - C: media global (prior) obtenida de cache/stats

    Intuición:
      - Con pocos votos, score → C (conservador).
      - Con muchos votos, score → R (evidencia fuerte).
    """
    if imdb_rating is None or imdb_votes is None:
        return None

    v = imdb_votes
    r = imdb_rating
    if v < 0:
        return None

    m_eff = max(0, int(m))
    if v + m_eff == 0:
        return None

    c = float(c_global)
    return (v / (v + m_eff)) * float(r) + (m_eff / (v + m_eff)) * c


# ============================================================================
# Main API
# ============================================================================

def compute_scoring(
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
    year: int | None = None,
    metacritic_score: int | None = None,
) -> ScoringDict:
    """
    Calcula la decisión final y devuelve ScoringDict:

    {
      "decision": "KEEP" | "DELETE" | "MAYBE" | "UNKNOWN",
      "reason": "...explicación humana...",
      "rule":   "ETIQUETA_ESTABLE",
      "inputs": {...snapshot...},
    }

    Reglas (resumen):
    0) Sin señales (ni IMDb ni RT ni Metacritic): UNKNOWN.
    1) Con score bayesiano:
        - >= bayes_keep_thr   -> KEEP_BAYES
        - <= bayes_delete_thr -> DELETE_BAYES
        - entre ambos         -> MAYBE_BAYES_MIDDLE
    2) RT alta puede boost a KEEP (si NO estamos en DELETE bayes fuerte).
    3) RT muy baja puede:
        - Confirmar DELETE si ya es DELETE por bayes.
        - Desempatar MAYBE cerca del delete (margen).
    4) Si no hay bayes pero hay IMDb+votos: fallbacks clásicos.
    5) Metacritic: solo añade contexto al reason (nunca cambia decision).

    Nota clave:
    - bayes_delete_thr se acota por BAYES_DELETE_MAX_SCORE para evitar “delete agresivo”.
    """
    # ------------------------------------------------------------------
    # Normalización defensiva de entradas (importante para lazy OMDb)
    # ------------------------------------------------------------------
    imdb_rating_f = _safe_float(imdb_rating)
    imdb_votes_i = _safe_int(imdb_votes)
    rt_score_i = _safe_int(rt_score)
    meta_i = _safe_int(metacritic_score)
    year_i = _safe_int(year)

    inputs: ScoringDict = {
        "imdb_rating": imdb_rating_f,
        "imdb_votes": imdb_votes_i,
        "rt_score": rt_score_i,
        "year": year_i,
        "metacritic_score": meta_i,
        # score_bayes SIEMPRE presente (consistencia en reporting)
        "score_bayes": None,
    }

    # ------------------------------------------------------------------
    # 0) Caso sin datos suficientes (incluye el caso típico lazy: todo None)
    # ------------------------------------------------------------------
    if imdb_rating_f is None and imdb_votes_i is None and rt_score_i is None and meta_i is None:
        return {
            "decision": "UNKNOWN",
            "reason": "Sin datos suficientes (IMDb/RT/Metacritic vacíos).",
            "rule": "NO_DATA",
            "inputs": inputs,
        }

    # Si solo hay Metacritic (sin IMDb/RT), seguimos siendo conservadores.
    if imdb_rating_f is None and imdb_votes_i is None and rt_score_i is None and meta_i is not None:
        return {
            "decision": "UNKNOWN",
            "reason": f"Solo hay Metacritic={meta_i}; sin IMDb/RT no es suficiente para decidir.",
            "rule": "META_ONLY",
            "inputs": inputs,
        }

    # ------------------------------------------------------------------
    # Umbrales bayesianos (auto-calibración)
    # ------------------------------------------------------------------
    # Robustez: cualquier fallo en stats/config degrada a valores seguros.
    try:
        bayes_keep_thr = float(get_auto_keep_rating_threshold())
    except Exception:
        bayes_keep_thr = float(IMDB_KEEP_MIN_RATING)

    try:
        bayes_delete_thr = float(get_auto_delete_rating_threshold())
    except Exception:
        bayes_delete_thr = float(IMDB_DELETE_MAX_RATING)

    bayes_delete_thr = min(bayes_delete_thr, float(BAYES_DELETE_MAX_SCORE))

    # Parámetros del prior bayesiano
    try:
        m_dynamic = int(get_votes_threshold_for_year(year_i))
    except Exception:
        m_dynamic = 0

    try:
        c_global = float(get_global_imdb_mean_from_cache())
    except Exception:
        c_global = 6.5  # fallback conservador razonable

    inputs["m_dynamic"] = m_dynamic
    inputs["c_global"] = c_global
    inputs["bayes_keep_thr"] = bayes_keep_thr
    inputs["bayes_delete_thr"] = bayes_delete_thr

    # ------------------------------------------------------------------
    # 1) Score bayesiano (si es posible) + decisión preliminar
    # ------------------------------------------------------------------
    bayes_score = _compute_bayes_score(
        imdb_rating=imdb_rating_f,
        imdb_votes=imdb_votes_i,
        m=m_dynamic,
        c_global=c_global,
    )
    inputs["score_bayes"] = bayes_score

    preliminary_decision: str | None = None
    preliminary_rule: str | None = None
    preliminary_reason: str | None = None

    if bayes_score is not None:
        # Estabilidad: clamp por si stats raros devolvieran valores extraños
        bayes_score = _clamp_float(bayes_score, lo=0.0, hi=10.0)

        if bayes_score >= bayes_keep_thr:
            preliminary_decision = "KEEP"
            preliminary_rule = "KEEP_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≥ KEEP={bayes_keep_thr:.2f} "
                f"(R={imdb_rating_f}, v={imdb_votes_i}, m={m_dynamic}, C={c_global:.2f})."
            )
        elif bayes_score <= bayes_delete_thr:
            preliminary_decision = "DELETE"
            preliminary_rule = "DELETE_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≤ DELETE={bayes_delete_thr:.2f} "
                f"(R={imdb_rating_f}, v={imdb_votes_i}, m={m_dynamic}, C={c_global:.2f})."
            )
        else:
            preliminary_decision = "MAYBE"
            preliminary_rule = "MAYBE_BAYES_MIDDLE"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} entre DELETE={bayes_delete_thr:.2f} "
                f"y KEEP={bayes_keep_thr:.2f} (evidencia intermedia)."
            )

    # ------------------------------------------------------------------
    # 2) RT alta: boost a KEEP (si no hay DELETE bayes fuerte)
    # ------------------------------------------------------------------
    if (
        rt_score_i is not None
        and imdb_rating_f is not None
        and rt_score_i >= int(RT_KEEP_MIN_SCORE)
        and imdb_rating_f >= float(IMDB_KEEP_MIN_RATING_WITH_RT)
    ):
        bayes_is_strong_delete = bayes_score is not None and bayes_score <= bayes_delete_thr
        if not bayes_is_strong_delete:
            return {
                "decision": "KEEP",
                "reason": (
                    f"RT={rt_score_i}% e imdbRating={imdb_rating_f} superan umbrales "
                    f"RT_KEEP_MIN_SCORE={RT_KEEP_MIN_SCORE} e "
                    f"IMDB_KEEP_MIN_RATING_WITH_RT={IMDB_KEEP_MIN_RATING_WITH_RT}; "
                    "RT refuerza la opinión positiva del público."
                ),
                "rule": "KEEP_RT_BOOST",
                "inputs": inputs,
            }

    # ------------------------------------------------------------------
    # 3) RT muy baja: confirma DELETE o desempata MAYBE cerca del delete
    # ------------------------------------------------------------------
    if rt_score_i is not None and rt_score_i <= int(RT_DELETE_MAX_SCORE) and bayes_score is not None:
        if preliminary_decision == "DELETE":
            return {
                "decision": "DELETE",
                "reason": (
                    f"{preliminary_reason or ''} Además RT={rt_score_i}% ≤ "
                    f"RT_DELETE_MAX_SCORE={RT_DELETE_MAX_SCORE}, lo que refuerza el borrado."
                ).strip(),
                "rule": "DELETE_BAYES_RT_CONFIRMED",
                "inputs": inputs,
            }

        if preliminary_decision == "MAYBE" and bayes_score <= (bayes_delete_thr + _RT_TIEBREAK_BAYES_MARGIN):
            return {
                "decision": "DELETE",
                "reason": (
                    f"score_bayes={bayes_score:.2f} cercano a DELETE={bayes_delete_thr:.2f} "
                    f"y RT={rt_score_i}% muy baja (≤ {RT_DELETE_MAX_SCORE}); el público es claramente negativo."
                ),
                "rule": "DELETE_RT_TIEBREAKER",
                "inputs": inputs,
            }

    # ------------------------------------------------------------------
    # 4) Sin bayes (None) pero con IMDb: fallbacks clásicos (rating + votos)
    # ------------------------------------------------------------------
    if bayes_score is None and imdb_rating_f is not None and imdb_votes_i is not None:
        dynamic_votes_needed = max(0, int(m_dynamic))

        if dynamic_votes_needed > 0 and imdb_rating_f >= float(IMDB_KEEP_MIN_RATING) and imdb_votes_i >= dynamic_votes_needed:
            return {
                "decision": "KEEP",
                "reason": (
                    "Sin score bayesiano, pero IMDb es alto con suficientes votos para su antigüedad: "
                    f"imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i} (mínimo por año={dynamic_votes_needed})."
                ),
                "rule": "KEEP_IMDB_FALLBACK",
                "inputs": inputs,
            }

        if dynamic_votes_needed > 0 and imdb_rating_f <= float(IMDB_DELETE_MAX_RATING) and imdb_votes_i >= dynamic_votes_needed:
            return {
                "decision": "DELETE",
                "reason": (
                    "Sin score bayesiano; IMDb muy bajo con suficientes votos para su antigüedad: "
                    f"imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i} (mínimo por año={dynamic_votes_needed})."
                ),
                "rule": "DELETE_IMDB_FALLBACK",
                "inputs": inputs,
            }

    # ------------------------------------------------------------------
    # 5) Si hay decisión preliminar por BAYES: devolverla (Metacritic solo refuerza reason)
    # ------------------------------------------------------------------
    if preliminary_decision is not None:
        reason = preliminary_reason or "Decisión derivada del score bayesiano."

        if meta_i is not None:
            if preliminary_decision == "KEEP" and meta_i >= int(METACRITIC_KEEP_MIN_SCORE):
                reason += f" La crítica también es favorable (Metacritic={meta_i})."
            elif preliminary_decision == "DELETE" and meta_i <= int(METACRITIC_DELETE_MAX_SCORE):
                reason += f" La crítica también es muy negativa (Metacritic={meta_i})."

        return {
            "decision": _decision_or_unknown(preliminary_decision),
            "reason": reason,
            "rule": preliminary_rule or "BAYES_GENERIC",
            "inputs": inputs,
        }

    # ------------------------------------------------------------------
    # 6) Sin bayes y sin reglas fuertes: MAYBE / UNKNOWN con explicación
    # ------------------------------------------------------------------
    # Si hay algo de señal (IMDb rating o RT) pero no es concluyente
    if imdb_rating_f is not None or rt_score_i is not None:
        # Pocos votos => conservador
        if imdb_votes_i is None or imdb_votes_i < int(IMDB_MIN_VOTES_FOR_KNOWN):
            return {
                "decision": "MAYBE",
                "reason": (
                    "Datos incompletos: hay señales, pero con pocos votos IMDb "
                    f"(imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i}, rt_score={rt_score_i})."
                ),
                "rule": "MAYBE_LOW_INFO",
                "inputs": inputs,
            }

        return {
            "decision": "MAYBE",
            "reason": (
                "No se puede clasificar claramente en KEEP/DELETE con las reglas actuales "
                f"(imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i}, rt_score={rt_score_i}, metacritic={meta_i})."
            ),
            "rule": "MAYBE_FALLBACK",
            "inputs": inputs,
        }

    # Último fallback
    return {
        "decision": "UNKNOWN",
        "reason": (
            "Información parcial no interpretable (por ejemplo, Metacritic sin IMDb/RT, o valores inválidos)."
        ),
        "rule": "UNKNOWN_PARTIAL",
        "inputs": inputs,
    }