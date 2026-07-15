import math

import pytest

from app.services.v21_pipeline import determine_final_decision


@pytest.mark.parametrize(("score", "expected"), [
    (0.0, "REAL"), (0.39, "REAL"), (0.444902, "REAL"),
    (0.499, "REAL"), (0.499999, "REAL"), (0.5, "MENCURIGAKAN"),
    (0.500001, "DEEPFAKE"), (0.8, "DEEPFAKE"), (1.0, "DEEPFAKE"),
])
def test_threshold_boundaries(score, expected):
    assert determine_final_decision(score) == expected


def test_tolerance_is_only_one_e_minus_nine():
    assert determine_final_decision(0.5000000001) == "MENCURIGAKAN"
    assert determine_final_decision(0.500000002) == "DEEPFAKE"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_non_finite_score_is_rejected(score):
    with pytest.raises(ValueError):
        determine_final_decision(score)


def test_out_of_range_values_are_clamped_consistently():
    assert determine_final_decision(-0.001) == "REAL"
    assert determine_final_decision(1.001) == "DEEPFAKE"

