from backend import scoring


# ---------------------------------------------------------
# _compute_bayes_score (helper interno)
# ---------------------------------------------------------


def test__compute_bayes_score_edge_cases() -> None:
    f = scoring._compute_bayes_score
    # Falta rating o votos
    assert f(None, 100, 5, 6.0) is None
    assert f(7.0, None, 5, 6.0) is None
    # votos negativos -> None
    assert f(7.0, -10, 5, 6.0) is None
    # v + m == 0 -> None
    assert f(7.0, 0, 0, 6.0) is None
    # caso normal
    val = f(8.0, 100, 10, 6.0)
    assert isinstance(val, float)


# ---------------------------------------------------------
# compute_scoring: casos básicos
# ---------------------------------------------------------


def test_compute_scoring_basic_keep(monkeypatch) -> None:
    # Entorno determinista: fijamos umbrales y media global
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 5)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.0)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 6.5)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.0)

    # Peli con buen rating y muchos votos → KEEP por bayes/umbrales
    res = scoring.compute_scoring(
        imdb_rating=8.5,
        imdb_votes=5000,
        rt_score=85,
        year=2015,
    )
    assert isinstance(res, dict)
    assert res["decision"] == "KEEP"
    assert "inputs" in res


def test_compute_scoring_no_data_unknown() -> None:
    res = scoring.compute_scoring(
        imdb_rating=None,
        imdb_votes=None,
        rt_score=None,
        year=None,
    )
    assert res["decision"] == "UNKNOWN"
    assert res["rule"] == "NO_DATA"


# ---------------------------------------------------------
# compute_scoring: RT como refuerzo positivo / negativo
# ---------------------------------------------------------


def test_compute_scoring_rt_boost_keep(monkeypatch) -> None:
    # Ajustamos entorno para que bayes no sea un DELETE fuerte,
    # pero RT alta + buen rating suban a KEEP_RT_BOOST.
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 50)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.0)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.5)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.0)

    res = scoring.compute_scoring(
        imdb_rating=7.0,
        imdb_votes=1000,
        rt_score=90,  # muy alto, ≥ RT_KEEP_MIN_SCORE
        year=2018,
    )

    assert res["decision"] == "KEEP"
    # Esta regla específica debe activarse cuando RT refuerza
    assert res["rule"] == "KEEP_RT_BOOST"


def test_compute_scoring_rt_low_confirms_delete(monkeypatch) -> None:
    # Forzamos un bayes bajo + RT muy baja
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 5.0)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.5)

    # Rating bajo y pocos votos → bayes cercano/bajo al umbral de delete
    res = scoring.compute_scoring(
        imdb_rating=3.5,
        imdb_votes=200,
        rt_score=10,  # RT muy baja
        year=2005,
    )

    assert res["decision"] == "DELETE"
    # Puede ser DELETE_BAYES_RT_CONFIRMED o DELETE_RT_TIEBREAKER
    assert res["rule"] in {
        "DELETE_BAYES_RT_CONFIRMED",
        "DELETE_RT_TIEBREAKER",
        "DELETE_BAYES",
    }


# ---------------------------------------------------------
# compute_scoring: fallbacks cuando bayes no se puede calcular
# ---------------------------------------------------------


def test_compute_scoring_fallback_keep_when_no_bayes(monkeypatch) -> None:
    # Forzamos que _compute_bayes_score devuelva None siempre
    monkeypatch.setattr(scoring, "_compute_bayes_score", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 1000)
    # Ajustamos umbrales clásicos
    monkeypatch.setattr(scoring, "IMDB_KEEP_MIN_RATING", 7.0, raising=False)
    monkeypatch.setattr(scoring, "IMDB_DELETE_MAX_RATING", 5.0, raising=False)

    res = scoring.compute_scoring(
        imdb_rating=7.5,
        imdb_votes=2000,  # ≥ dynamic_votes_needed
        rt_score=None,
        year=2010,
    )

    assert res["decision"] == "KEEP"
    assert res["rule"] in {"KEEP_IMDB_FALLBACK", "KEEP_BAYES", "KEEP_RT_BOOST"}


def test_compute_scoring_fallback_delete_when_no_bayes(monkeypatch) -> None:
    monkeypatch.setattr(scoring, "_compute_bayes_score", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 500)
    monkeypatch.setattr(scoring, "IMDB_KEEP_MIN_RATING", 7.0, raising=False)
    monkeypatch.setattr(scoring, "IMDB_DELETE_MAX_RATING", 5.0, raising=False)

    res = scoring.compute_scoring(
        imdb_rating=4.0,
        imdb_votes=2000,
        rt_score=None,
        year=2000,
    )

    assert res["decision"] in {"DELETE", "MAYBE"}
    # En escenarios razonables debería activar el fallback clásico
    assert res["rule"] in {
        "DELETE_IMDB_FALLBACK",
        "DELETE_BAYES",
        "DELETE_RT_TIEBREAKER",
        "DELETE_BAYES_RT_CONFIRMED",
    }


def test_compute_scoring_maybe_low_info_few_votes(monkeypatch) -> None:
    # No bayes, rating presente pero muy pocos votos
    monkeypatch.setattr(scoring, "_compute_bayes_score", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "IMDB_MIN_VOTES_FOR_KNOWN", 100, raising=False)
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 50)

    res = scoring.compute_scoring(
        imdb_rating=7.0,
        imdb_votes=10,  # muy pocos votos
        rt_score=None,
        year=2020,
    )

    assert res["decision"] == "MAYBE"
    assert res["rule"] == "MAYBE_LOW_INFO"


# ---------------------------------------------------------
# compute_scoring: refuerzo Metacritic
# ---------------------------------------------------------


def test_compute_scoring_metacritic_reinforces_keep(monkeypatch) -> None:
    # Hacemos que bayes dé un KEEP claro
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.0)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.0)
    monkeypatch.setattr(scoring, "METACRITIC_KEEP_MIN_SCORE", 70, raising=False)

    res = scoring.compute_scoring(
        imdb_rating=8.0,
        imdb_votes=1000,
        rt_score=None,
        year=2012,
        metacritic_score=80,
    )

    assert res["decision"] == "KEEP"
    assert "Metacritic" in res["reason"]


def test_compute_scoring_metacritic_reinforces_delete(monkeypatch) -> None:
    # Bayes bien bajo → DELETE, metacritic baja debería reforzar
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 5.0)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.0)
    monkeypatch.setattr(scoring, "METACRITIC_DELETE_MAX_SCORE", 40, raising=False)

    res = scoring.compute_scoring(
        imdb_rating=3.5,
        imdb_votes=2000,
        rt_score=None,
        year=2005,
        metacritic_score=20,
    )

    assert res["decision"] == "DELETE"
    assert "Metacritic" in res["reason"]


# ---------------------------------------------------------
# compute_scoring: UNKNOWN_PARTIAL y MAYBE_FALLBACK
# ---------------------------------------------------------


def test_compute_scoring_unknown_partial_with_only_metacritic(monkeypatch) -> None:
    # Sin IMDb ni RT, solo metacritic y votos → UNKNOWN_PARTIAL
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)

    res = scoring.compute_scoring(
        imdb_rating=None,
        imdb_votes=1000,
        rt_score=None,
        year=2010,
        metacritic_score=80,
    )

    assert res["decision"] == "UNKNOWN"
    assert res["rule"] == "UNKNOWN_PARTIAL"


def test_compute_scoring_maybe_fallback_with_enough_votes(monkeypatch) -> None:
    # IMDb rating, suficientes votos, pero sin bayes fuerte ni reglas claras
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.0)
    # Umbrales muy extremos para forzar un MAYBE
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 9.5)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 2.0)

    res = scoring.compute_scoring(
        imdb_rating=6.0,
        imdb_votes=500,
        rt_score=None,
        year=2010,
    )

    assert res["decision"] == "MAYBE"
    assert res["rule"] in {"MAYBE_BAYES_MIDDLE", "MAYBE_FALLBACK", "MAYBE_LOW_INFO"}


# ---------------------------------------------------------
# decide_action (envoltorio)
# ---------------------------------------------------------


def test_decide_action_tuple(monkeypatch) -> None:
    # Entorno determinista similar al primer test
    monkeypatch.setattr(scoring, "get_votes_threshold_for_year", lambda y: 10)
    monkeypatch.setattr(scoring, "get_global_imdb_mean_from_cache", lambda: 6.5)
    monkeypatch.setattr(scoring, "get_auto_keep_rating_threshold", lambda: 7.0)
    monkeypatch.setattr(scoring, "get_auto_delete_rating_threshold", lambda: 4.0)

    d, reason = scoring.decide_action(
        imdb_rating=7.5,
        imdb_votes=1500,
        rt_score=60,
        year=2010,
    )
    assert isinstance(d, str)
    assert isinstance(reason, str)
    assert d in ("KEEP", "MAYBE", "DELETE", "UNKNOWN")