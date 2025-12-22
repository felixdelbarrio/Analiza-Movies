from __future__ import annotations

"""
scoring.py

Motor de decisión (KEEP / DELETE / MAYBE / UNKNOWN) basado en señales de calidad
(IMDb, Rotten Tomatoes, Metacritic) con un “score bayesiano” como regla principal.

Objetivo
- Dar una decisión estable, explicable (reason + rule) y trazable (inputs).
- Evitar falsos positivos de borrado en pelis con pocos votos (ruido estadístico).
- Mantener la lógica “conservadora”: Metacritic SOLO refuerza, nunca salva.

Modelo (alto nivel)
1) Regla principal: score bayesiano IMDb (rating + votos + prior global).
2) Señales suaves del público: Rotten Tomatoes (boost o desempate).
3) Señal de crítica: Metacritic (refuerzo únicamente en explicación).
4) Fallbacks: si no hay score bayesiano, se aplican reglas clásicas (rating/votos).
5) Si no hay suficiente información: MAYBE / UNKNOWN.

Convenciones de salida (ScoringDict)
- decision: "KEEP" | "DELETE" | "MAYBE" | "UNKNOWN"
- rule: etiqueta estable para auditoría / tests / troubleshooting
- reason: explicación humana (para CSV/dashboard)
- inputs: snapshot de señales/umbrales usados (para debug y reproducibilidad)
"""

from typing import Final

from backend.config import (
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

ScoringDict = dict[str, object]

# Pequeño margen usado en tiebreakers para evitar “flip-flop” cerca del umbral
_RT_TIEBREAK_BAYES_MARGIN: Final[float] = 0.30


def _safe_int(value: object) -> int | None:
    """Convierte a int de forma defensiva; devuelve None si no es posible."""
    try:
        if value is None:
            return None
        v = int(value)  # type: ignore[arg-type]
        return v
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    """Convierte a float de forma defensiva; devuelve None si no es posible."""
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _compute_bayes_score(
    imdb_rating: float | None,
    imdb_votes: int | None,
    m: int,
    c_global: float,
) -> float | None:
    """
    Calcula el score bayesiano IMDb:

        score_bayes = (v / (v + m)) * R + (m / (v + m)) * C

    Donde:
      - R: imdb_rating (rating observado)
      - v: imdb_votes (votos observados)
      - m: “prior strength”: votos mínimos/esperados según antigüedad
      - C: media global (prior): promedio global de IMDb (cache)

    Intuición:
    - Con pocos votos (v pequeño), el score se acerca a C (más conservador).
    - Con muchos votos, el score se acerca a R.

    Devuelve None si no se puede calcular (datos inválidos o insuficientes).
    """
    if imdb_rating is None or imdb_votes is None:
        return None

    v = _safe_int(imdb_votes)
    r = _safe_float(imdb_rating)
    if v is None or r is None:
        return None
    if v < 0:
        return None

    m = max(0, int(m))
    if v + m == 0:
        return None

    c = float(c_global)
    return (v / (v + m)) * r + (m / (v + m)) * c


def compute_scoring(
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
    year: int | None = None,
    metacritic_score: int | None = None,
) -> ScoringDict:
    """
    Calcula la decisión final y devuelve ScoringDict (decision, reason, rule, inputs).

    Reglas (resumen):
    - Si no hay IMDb ni RT: UNKNOWN.
    - Si hay score bayesiano:
        * >= bayes_keep_thr   -> KEEP_BAYES
        * <= bayes_delete_thr -> DELETE_BAYES
        * entre ambos         -> MAYBE_BAYES_MIDDLE
    - RT puede:
        * Boost a KEEP si RT alta y IMDb también alto (y no hay DELETE bayes fuerte).
        * Confirmar DELETE o desempatar MAYBE cerca del delete.
    - Metacritic solo añade contexto en el reason (nunca cambia decision).

    Nota sobre umbrales:
    - bayes_keep_thr y bayes_delete_thr vienen de backend.stats (auto calibración).
    - bayes_delete_thr se acota con BAYES_DELETE_MAX_SCORE para evitar “delete agresivo”.
    """
    # Normalización defensiva (por si llega algo como "7.1" / "12000")
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
        # score_bayes SIEMPRE presente para reporting uniforme
        "score_bayes": None,
    }

    # ----------------------------------------------------
    # 0) Caso sin datos suficientes
    # ----------------------------------------------------
    if imdb_rating_f is None and imdb_votes_i is None and rt_score_i is None:
        return {
            "decision": "UNKNOWN",
            "reason": "Sin datos suficientes de OMDb (IMDb y RT vacíos).",
            "rule": "NO_DATA",
            "inputs": inputs,
        }

    # ----------------------------------------------------
    # Umbrales efectivos para score bayesiano
    # ----------------------------------------------------
    bayes_keep_thr: float = float(get_auto_keep_rating_threshold())
    bayes_delete_thr: float = float(get_auto_delete_rating_threshold())
    bayes_delete_thr = min(bayes_delete_thr, float(BAYES_DELETE_MAX_SCORE))

    # Parámetros del score bayesiano
    m_dynamic: int = int(get_votes_threshold_for_year(year_i))
    c_global: float = float(get_global_imdb_mean_from_cache())

    inputs["m_dynamic"] = m_dynamic
    inputs["c_global"] = c_global
    inputs["bayes_keep_thr"] = bayes_keep_thr
    inputs["bayes_delete_thr"] = bayes_delete_thr

    # ----------------------------------------------------
    # 1) Score bayesiano (si es posible) + decisión preliminar
    # ----------------------------------------------------
    bayes_score: float | None = _compute_bayes_score(
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
        if bayes_score >= bayes_keep_thr:
            preliminary_decision = "KEEP"
            preliminary_rule = "KEEP_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≥ umbral KEEP={bayes_keep_thr:.2f} "
                f"(R={imdb_rating_f}, v={imdb_votes_i}, m={m_dynamic}, C={c_global:.2f})."
            )
        elif bayes_score <= bayes_delete_thr:
            preliminary_decision = "DELETE"
            preliminary_rule = "DELETE_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≤ umbral DELETE={bayes_delete_thr:.2f} "
                f"(R={imdb_rating_f}, v={imdb_votes_i}, m={m_dynamic}, C={c_global:.2f})."
            )
        else:
            preliminary_decision = "MAYBE"
            preliminary_rule = "MAYBE_BAYES_MIDDLE"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} entre umbral DELETE={bayes_delete_thr:.2f} "
                f"y KEEP={bayes_keep_thr:.2f} (evidencia intermedia)."
            )

    # ----------------------------------------------------
    # 2) RT alta: boost a KEEP (si no estamos en DELETE bayes fuerte)
    # ----------------------------------------------------
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
                    "RT refuerza una opinión positiva del público."
                ),
                "rule": "KEEP_RT_BOOST",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 3) RT muy baja: refuerza DELETE / desempata MAYBE cerca del delete
    # ----------------------------------------------------
    if (
        rt_score_i is not None
        and rt_score_i <= int(RT_DELETE_MAX_SCORE)
        and imdb_rating_f is not None
        and bayes_score is not None
        and bayes_score <= bayes_keep_thr
    ):
        if preliminary_decision == "DELETE":
            return {
                "decision": "DELETE",
                "reason": (
                    f"{preliminary_reason or ''} Además RT={rt_score_i}% ≤ "
                    f"RT_DELETE_MAX_SCORE={RT_DELETE_MAX_SCORE}, lo que refuerza la decisión de borrar "
                    "(público muy negativo)."
                ).strip(),
                "rule": "DELETE_BAYES_RT_CONFIRMED",
                "inputs": inputs,
            }

        if preliminary_decision == "MAYBE" and bayes_score <= (bayes_delete_thr + _RT_TIEBREAK_BAYES_MARGIN):
            return {
                "decision": "DELETE",
                "reason": (
                    f"score_bayes={bayes_score:.2f} cercano al umbral DELETE={bayes_delete_thr:.2f} "
                    f"y RT={rt_score_i}% muy baja (≤ {RT_DELETE_MAX_SCORE}); el público es claramente negativo."
                ),
                "rule": "DELETE_RT_TIEBREAKER",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 4) Sin bayes (None) pero con IMDb: fallbacks clásicos (rating + votos)
    # ----------------------------------------------------
    if bayes_score is None and imdb_rating_f is not None and imdb_votes_i is not None:
        dynamic_votes_needed: int = m_dynamic

        if (
            dynamic_votes_needed > 0
            and imdb_rating_f >= float(IMDB_KEEP_MIN_RATING)
            and imdb_votes_i >= dynamic_votes_needed
        ):
            return {
                "decision": "KEEP",
                "reason": (
                    "No se pudo calcular score bayesiano, pero imdbRating y votos son altos para su antigüedad: "
                    f"imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i} (mínimo por año={dynamic_votes_needed})."
                ),
                "rule": "KEEP_IMDB_FALLBACK",
                "inputs": inputs,
            }

        if (
            dynamic_votes_needed > 0
            and imdb_rating_f <= float(IMDB_DELETE_MAX_RATING)
            and imdb_votes_i >= dynamic_votes_needed
        ):
            return {
                "decision": "DELETE",
                "reason": (
                    "No se pudo calcular score bayesiano; rating IMDb muy bajo con muchos votos para su antigüedad: "
                    f"imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i}, mínimo por año={dynamic_votes_needed}."
                ),
                "rule": "DELETE_IMDB_FALLBACK",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 5) Si hay decisión preliminar por BAYES: devolverla (Metacritic solo refuerza reason)
    # ----------------------------------------------------
    if preliminary_decision is not None:
        reason = preliminary_reason or "Decisión derivada del score bayesiano."
        if meta_i is not None:
            if preliminary_decision == "KEEP" and meta_i >= int(METACRITIC_KEEP_MIN_SCORE):
                reason += f" La crítica especializada también es favorable (Metacritic={meta_i})."
            elif preliminary_decision == "DELETE" and meta_i <= int(METACRITIC_DELETE_MAX_SCORE):
                reason += f" La crítica especializada también es muy negativa (Metacritic={meta_i})."

        return {
            "decision": preliminary_decision,
            "reason": reason,
            "rule": preliminary_rule or "BAYES_GENERIC",
            "inputs": inputs,
        }

    # ----------------------------------------------------
    # 6) Sin bayes y sin reglas fuertes: MAYBE / UNKNOWN con explicación
    # ----------------------------------------------------
    if imdb_rating_f is not None or rt_score_i is not None:
        if imdb_votes_i is None or imdb_votes_i < int(IMDB_MIN_VOTES_FOR_KNOWN):
            return {
                "decision": "MAYBE",
                "reason": (
                    "Datos incompletos: rating disponible pero con pocos votos IMDb "
                    f"(imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i})."
                ),
                "rule": "MAYBE_LOW_INFO",
                "inputs": inputs,
            }

        return {
            "decision": "MAYBE",
            "reason": (
                "No se ha podido clasificar claramente en KEEP/DELETE con las reglas actuales "
                f"(imdbRating={imdb_rating_f}, imdbVotes={imdb_votes_i}, rt_score={rt_score_i}, metacritic={meta_i})."
            ),
            "rule": "MAYBE_FALLBACK",
            "inputs": inputs,
        }

    return {
        "decision": "UNKNOWN",
        "reason": (
            "Solo hay información parcial (p.ej. RT o Metacritic sin IMDb) y no es suficiente "
            "para tomar una decisión segura."
        ),
        "rule": "UNKNOWN_PARTIAL",
        "inputs": inputs,
    }