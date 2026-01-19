import math

from frontend.tabs import charts_shared


def test_format_num_handles_none_nan():
    assert charts_shared._format_num(None) == "N/A"
    assert charts_shared._format_num(float("nan")) == "N/A"
    assert charts_shared._format_num(7.126) == "7.1"
    assert charts_shared._format_num(7.126, ".2f") == "7.13"


def test_corr_strength_thresholds():
    assert charts_shared._corr_strength(0.75) == "alta"
    assert charts_shared._corr_strength(-0.7) == "alta"
    assert charts_shared._corr_strength(0.55) == "moderada"
    assert charts_shared._corr_strength(0.3) == "baja"
    assert charts_shared._corr_strength(0.05) == "muy baja"


def test_weighted_revision_uses_weights():
    score = charts_shared._weighted_revision(2, 3)
    expected = 2 * charts_shared.DELETE_WEIGHT + 3 * charts_shared.MAYBE_WEIGHT
    assert math.isclose(score, expected)
