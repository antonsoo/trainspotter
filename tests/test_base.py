from __future__ import annotations

import numpy as np

from trainspotter.detectors.base import ols_slope


def test_ols_slope_detects_a_clear_upward_trend() -> None:
    x = np.arange(50, dtype=float)
    y = 2.0 * x + 1.0
    slope, _se, p_value = ols_slope(x, y)
    assert slope == 2.0
    assert p_value < 0.01


def test_ols_slope_on_flat_noise_is_not_significant() -> None:
    rng = np.random.default_rng(1)
    x = np.arange(50, dtype=float)
    y = rng.normal(0, 0.01, size=50)
    _slope, _se, p_value = ols_slope(x, y)
    assert p_value > 0.05


def test_ols_slope_with_non_finite_values_does_not_warn_or_crash() -> None:
    # A window overlapping a diverged run's NaN/Inf reading must not feed
    # numpy arithmetic that would raise RuntimeWarnings (or return NaN
    # itself, which a caller would then have to special-case).
    x = np.arange(10, dtype=float)
    y = np.array([1.0, 2.0, 3.0, float("inf"), 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    with np.errstate(all="raise"):
        slope, se, p_value = ols_slope(x, y)
    assert slope == 0.0
    assert se == float("inf")
    assert p_value == 1.0
