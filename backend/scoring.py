from __future__ import annotations

from typing import Final

from backend.config import (
    BAYES_DELETE_MAX_SCORE,
    IMDB_DELETE_MAX_RATING,
    IMDB_KEEP_MIN_RATING,
    IMDB_KEEP_MIN_RATING_WITH_RT,
    IMDB_MIN_VOTES_FOR_KNOWN,
    RT_DELETE_MAX_SCORE,
    RT_KEEP_MIN_SCORE,
    METACRITIC_KEEP_MIN_SCORE,
    METACRITIC_DELETE_MAX_SCORE,
    get_votes_threshold_for_year,
)
from backend.stats import (
    get_global_imdb_mean_from_cache,
    get_auto_keep_rating_threshold,
    get_auto_delete_rating_threshold,
)

ScoringDict = dict[str, object]


def _compute_bayes_score(
    imdb_rating: float | None,
    imdb_votes: int | None,
    m: int,
    c_global: float,
) -> float | None:
    """
    score_bayes = (v / (v + m)) * R + (m / (v + m)) * C

      - R: imdb_rating
      - v: imdb_votes
      - m: número mínimo de votos (según antigüedad)
      - C: media global (omdb_cache o BAYES_GLOBAL_MEAN_DEFAULT)

    Devuelve None si no se puede calcular.
    """
    if imdb_rating is None or imdb_votes is None:
        return None

    try:
        v = int(imdb_votes)
        if v < 0:
            return None
    except (TypeError, ValueError):
        return None

    if m < 0:
        m = 0

    if v + m == 0:
        return None

    r = float(imdb_rating)
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
    Calcula la decisión de KEEP / DELETE / MAYBE / UNKNOWN y devuelve
    un objeto de scoring enriquecido:

      - decision: str
      - reason: str
      - rule: str
      - inputs: dict con imdb_rating, imdb_votes, rt_score, year, score_bayes...

    Modelo:

      1) El score bayesiano es la regla principal:
         - score_bayes >= bayes_keep_thr   → KEEP_BAYES
         - score_bayes <= bayes_delete_thr → DELETE_BAYES
         - en medio                        → MAYBE_BAYES_MIDDLE

      2) Rotten Tomatoes actúa como señal suave (público también):
         - RT alta puede subir MAYBE→KEEP si no es un DELETE claro por bayes.
         - RT muy baja refuerza DELETE o rompe empates cerca del umbral.

      3) Metacritic (crítica especializada) SOLO refuerza:
         - Si decisión=KEEP y Metacritic ≥ METACRITIC_KEEP_MIN_SCORE → se añade a la razón.
         - Si decisión=DELETE y Metacritic ≤ METACRITIC_DELETE_MAX_SCORE → se añade a la razón.
         - Nunca convierte una peli mala del público en KEEP.

      4) Si no hay datos suficientes para bayes, se usan fallbacks por rating/votos.
    """
    inputs: ScoringDict = {
        "imdb_rating": imdb_rating,
        "imdb_votes": imdb_votes,
        "rt_score": rt_score,
        "year": year,
        "metacritic_score": metacritic_score,
    }

    # ----------------------------------------------------
    # Caso sin datos suficientes de ningún tipo
    # ----------------------------------------------------
    if imdb_rating is None and imdb_votes is None and rt_score is None:
        return {
            "decision": "UNKNOWN",
            "reason": "Sin datos suficientes de OMDb (IMDb y RT vacíos).",
            "rule": "NO_DATA",
            "inputs": inputs,
        }

    # ----------------------------------------------------
    # Umbrales efectivos para score bayesiano
    # ----------------------------------------------------
    bayes_keep_thr: float = get_auto_keep_rating_threshold()
    bayes_delete_thr: float = get_auto_delete_rating_threshold()
    # El umbral de delete no puede superar el máximo configurable
    bayes_delete_thr = min(bayes_delete_thr, BAYES_DELETE_MAX_SCORE)

    # ----------------------------------------------------
    # Cálculo del score bayesiano (si es posible)
    # ----------------------------------------------------
    bayes_score: float | None = None
    m_dynamic: int = get_votes_threshold_for_year(year)
    c_global: float = get_global_imdb_mean_from_cache()

    if imdb_rating is not None and imdb_votes is not None:
        bayes_score = _compute_bayes_score(
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
            m=m_dynamic,
            c_global=c_global,
        )
        inputs["score_bayes"] = bayes_score

    # Añadimos información contextual a inputs siempre (útil para logging/debug)
    inputs["m_dynamic"] = m_dynamic
    inputs["c_global"] = c_global
    inputs["bayes_keep_thr"] = bayes_keep_thr
    inputs["bayes_delete_thr"] = bayes_delete_thr

    # ----------------------------------------------------
    # 1) DECISIÓN PRINCIPAL: BAYES
    # ----------------------------------------------------
    preliminary_decision: str | None = None
    preliminary_rule: str | None = None
    preliminary_reason: str | None = None

    if bayes_score is not None:
        if bayes_score >= bayes_keep_thr:
            preliminary_decision = "KEEP"
            preliminary_rule = "KEEP_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≥ umbral KEEP={bayes_keep_thr:.2f} "
                f"(R={imdb_rating}, v={imdb_votes}, m={m_dynamic}, C={c_global:.2f})."
            )
        elif bayes_score <= bayes_delete_thr:
            preliminary_decision = "DELETE"
            preliminary_rule = "DELETE_BAYES"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} ≤ umbral DELETE={bayes_delete_thr:.2f} "
                f"(R={imdb_rating}, v={imdb_votes}, m={m_dynamic}, C={c_global:.2f})."
            )
        else:
            preliminary_decision = "MAYBE"
            preliminary_rule = "MAYBE_BAYES_MIDDLE"
            preliminary_reason = (
                f"score_bayes={bayes_score:.2f} entre umbral DELETE={bayes_delete_thr:.2f} "
                f"y KEEP={bayes_keep_thr:.2f} (evidencia intermedia)."
            )

    # ----------------------------------------------------
    # 2) REGLA SUAVE POSITIVA: RT alta puede subir a KEEP
    # ----------------------------------------------------
    if (
        rt_score is not None
        and imdb_rating is not None
        and rt_score >= RT_KEEP_MIN_SCORE
        and imdb_rating >= IMDB_KEEP_MIN_RATING_WITH_RT
    ):
        bayes_is_strong_delete = (
            bayes_score is not None and bayes_score <= bayes_delete_thr
        )

        if not bayes_is_strong_delete:
            return {
                "decision": "KEEP",
                "reason": (
                    f"RT={rt_score}% e imdbRating={imdb_rating} "
                    f"superan umbrales RT_KEEP_MIN_SCORE={RT_KEEP_MIN_SCORE} "
                    f"e IMDB_KEEP_MIN_RATING_WITH_RT={IMDB_KEEP_MIN_RATING_WITH_RT}; "
                    "RT refuerza una opinión positiva del público."
                ),
                "rule": "KEEP_RT_BOOST",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 3) REGLA SUAVE NEGATIVA: RT muy baja apoya DELETE
    # ----------------------------------------------------
    if (
        rt_score is not None
        and rt_score <= RT_DELETE_MAX_SCORE
        and imdb_rating is not None
        and bayes_score is not None
        and bayes_score <= bayes_keep_thr
    ):
        if preliminary_decision == "DELETE":
            return {
                "decision": "DELETE",
                "reason": (
                    f"{preliminary_reason or ''} Además RT={rt_score}% ≤ "
                    f"RT_DELETE_MAX_SCORE={RT_DELETE_MAX_SCORE}, "
                    "lo que refuerza la decisión de borrar (público muy negativo)."
                ).strip(),
                "rule": "DELETE_BAYES_RT_CONFIRMED",
                "inputs": inputs,
            }

        if preliminary_decision == "MAYBE" and bayes_score <= (
            bayes_delete_thr + 0.3
        ):
            return {
                "decision": "DELETE",
                "reason": (
                    f"score_bayes={bayes_score:.2f} cercano al umbral DELETE={bayes_delete_thr:.2f} "
                    f"y RT={rt_score}% muy baja (≤ {RT_DELETE_MAX_SCORE}); "
                    "el público es claramente negativo."
                ),
                "rule": "DELETE_RT_TIEBREAKER",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 4) FALLO EN BAYES → fallbacks por rating/votos clásicos
    # ----------------------------------------------------
    if bayes_score is None and imdb_rating is not None and imdb_votes is not None:
        dynamic_votes_needed: int = m_dynamic

        # 4.1 KEEP clásico
        if (
            dynamic_votes_needed > 0
            and imdb_rating >= IMDB_KEEP_MIN_RATING
            and imdb_votes >= dynamic_votes_needed
        ):
            return {
                "decision": "KEEP",
                "reason": (
                    "No se pudo calcular score bayesiano, pero imdbRating y votos son altos "
                    f"para su antigüedad: imdbRating={imdb_rating}, imdbVotes={imdb_votes} "
                    f"(mínimo por año={dynamic_votes_needed})."
                ),
                "rule": "KEEP_IMDB_FALLBACK",
                "inputs": inputs,
            }

        # 4.2 DELETE clásico
        if (
            dynamic_votes_needed > 0
            and imdb_rating <= IMDB_DELETE_MAX_RATING
            and imdb_votes >= dynamic_votes_needed
        ):
            return {
                "decision": "DELETE",
                "reason": (
                    "No se pudo calcular score bayesiano; rating IMDb muy bajo con muchos votos "
                    f"para su antigüedad: imdbRating={imdb_rating}, imdbVotes={imdb_votes}, "
                    f"mínimo por año={dynamic_votes_needed}."
                ),
                "rule": "DELETE_IMDB_FALLBACK",
                "inputs": inputs,
            }

    # ----------------------------------------------------
    # 5) Si tenemos decisión preliminar por BAYES, la usamos
    #     y dejamos que Metacritic SOLO refuerce la explicación
    # ----------------------------------------------------
    if preliminary_decision is not None:
        reason = preliminary_reason or "Decisión derivada del score bayesiano."
        meta = metacritic_score

        if meta is not None:
            if preliminary_decision == "KEEP" and meta >= METACRITIC_KEEP_MIN_SCORE:
                reason += (
                    f" La crítica especializada también es favorable "
                    f"(Metacritic={meta})."
                )
            elif (
                preliminary_decision == "DELETE"
                and meta <= METACRITIC_DELETE_MAX_SCORE
            ):
                reason += (
                    f" La crítica especializada también es muy negativa "
                    f"(Metacritic={meta})."
                )

        return {
            "decision": preliminary_decision,
            "reason": reason,
            "rule": preliminary_rule or "BAYES_GENERIC",
            "inputs": inputs,
        }

    # ----------------------------------------------------
    # 6) Sin bayes y sin reglas fuertes → MAYBE / UNKNOWN más explicado
    # ----------------------------------------------------
    if imdb_rating is not None or rt_score is not None:
        if imdb_votes is None or imdb_votes < IMDB_MIN_VOTES_FOR_KNOWN:
            return {
                "decision": "MAYBE",
                "reason": (
                    "Datos incompletos: rating disponible pero con pocos votos IMDb "
                    f"(imdbRating={imdb_rating}, imdbVotes={imdb_votes})."
                ),
                "rule": "MAYBE_LOW_INFO",
                "inputs": inputs,
            }

        return {
            "decision": "MAYBE",
            "reason": (
                "No se ha podido clasificar claramente en KEEP/DELETE con las reglas "
                f"actuales (imdbRating={imdb_rating}, imdbVotes={imdb_votes}, "
                f"rt_score={rt_score}, metacritic={metacritic_score})."
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


def decide_action(
    imdb_rating: float | None,
    imdb_votes: int | None,
    rt_score: int | None,
    year: int | None = None,
    metacritic_score: int | None = None,
) -> tuple[str, str]:
    """
    Devuelve (decision, reason) usando compute_scoring.

    metacritic_score es opcional (0-100). Todas las llamadas anteriores
    que pasan solo (rating, votos, rt, year) siguen funcionando igual.
    """
    result = compute_scoring(
        imdb_rating=imdb_rating,
        imdb_votes=imdb_votes,
        rt_score=rt_score,
        year=year,
        metacritic_score=metacritic_score,
    )
    decision = result.get("decision")
    reason = result.get("reason")

    # Protección de tipos por si algo raro se cuela en el dict
    decision_str = str(decision) if decision is not None else "UNKNOWN"
    reason_str = str(reason) if reason is not None else ""
    return decision_str, reason_str