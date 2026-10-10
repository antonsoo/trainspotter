"""Independent window definitions, high-precision fits, and decision boundaries."""

from __future__ import annotations

import math
import statistics
from decimal import Decimal, localcontext

import numpy as np
import pytest

from trainspotter.detectors._rolling_slope import flat_slope_windows
from trainspotter.detectors.base import (
    centered_median,
    ols_slope,
    rolling_median_mad,
    trailing_median,
)


@pytest.mark.parametrize("window", [1, 2, 5, 21, 200, 201, 7000])
def test_medians_match_sorted_windows_and_do_not_mutate_history(window: int) -> None:
    rng = np.random.default_rng(42)
    values = rng.integers(-10, 30, 6000).astype(float)[::2]
    original = values.copy()
    values.flags.writeable = False
    trailing = [
        statistics.median(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))
    ]
    mad = [
        statistics.median(abs(v - trailing[i]) for v in values[max(0, i - window + 1) : i + 1])
        for i in range(len(values))
    ]
    centered = [
        statistics.median(values[max(0, i - window // 2) : i + window // 2 + 1])
        for i in range(len(values))
    ]
    actual_med, actual_mad = rolling_median_mad(values, window)
    np.testing.assert_array_equal(actual_med, trailing)
    np.testing.assert_array_equal(actual_mad, mad)
    np.testing.assert_array_equal(trailing_median(values, window), trailing)
    np.testing.assert_array_equal(centered_median(values, window), centered)
    np.testing.assert_array_equal(values, original)


def test_empty_and_nonfinite_median_windows_keep_their_original_meaning() -> None:
    empty = np.array([], dtype=float)
    assert len(centered_median(empty, 5)) == len(trailing_median(empty, 5)) == 0
    assert all(len(v) == 0 for v in rolling_median_mad(empty, 5))
    values = np.tile([1.0, math.nan, 2.0, math.inf, -math.inf, 3.0], 500)
    with np.errstate(invalid="ignore"):
        med = [np.median(values[max(0, i - 20) : i + 1]) for i in range(len(values))]
        mad = [
            np.median(np.abs(values[max(0, i - 20) : i + 1] - med[i])) for i in range(len(values))
        ]
        expected = [np.median(values[max(0, i - 2) : i + 3]) for i in range(len(values))]
        actual_med, actual_mad = rolling_median_mad(values, 21)
        np.testing.assert_array_equal(actual_med, med)
        np.testing.assert_array_equal(actual_mad, mad)
        np.testing.assert_array_equal(centered_median(values, 5), expected)


def decimal_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """80-digit normal equations on the exact stored doubles; no NumPy fit."""
    with localcontext() as context:
        context.prec = 80
        dx = [Decimal.from_float(float(v)) for v in x]
        dy = [Decimal.from_float(float(v)) for v in y]
        n = Decimal(len(dx))
        xx = n * sum(v * v for v in dx) - sum(dx) ** 2
        yy = n * sum(v * v for v in dy) - sum(dy) ** 2
        xy = n * sum(a * b for a, b in zip(dx, dy, strict=True)) - sum(dx) * sum(dy)
        if not xx:
            return 0.0, math.inf, 1.0
        slope = xy / xx
        se = ((yy - xy * xy / xx) / xx / (n - 2)).sqrt()
        p = math.erfc(abs(float(slope / se)) / math.sqrt(2)) if se else float(not slope)
        return float(slope), float(se), p


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e12, 1e15])
@pytest.mark.parametrize("scale", [1.0, 1e-150, 1e150])
def test_fits_match_high_precision_with_offsets_and_rescaling(offset: float, scale: float) -> None:
    rng = np.random.default_rng(731)
    x = np.cumsum(rng.integers(1, 8, 360)).astype(float) + offset
    y = (rng.normal(0, 0.2, 360) + offset) * scale
    window = 40
    mask = flat_slope_windows(x, y, window, 0.2, np.ones(len(y) - window + 1, dtype=bool))
    for i in range(0, len(mask), 17):
        expected = decimal_fit(x[i : i + window], y[i : i + window])
        actual = ols_slope(x[i : i + window], y[i : i + window])
        np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-11, atol=0)
        assert actual[2] == pytest.approx(expected[2], abs=1e-12)
        assert bool(mask[i]) == (expected[2] >= 0.2)


def test_threshold_neighbours_and_exact_equality_use_the_direct_decision() -> None:
    window = 101
    x = np.arange(window, dtype=float)
    dx = x - x.mean()
    noise = np.cos(x)
    noise -= noise.mean()
    noise -= np.dot(noise, dx) / np.dot(dx, dx) * dx
    noise /= np.linalg.norm(noise)
    critical = -statistics.NormalDist().inv_cdf(0.1)
    for relative in [-1e-6, -1e-12, 0.0, 1e-12, 1e-6]:
        y = noise + critical * (1 + relative) * dx / math.sqrt(np.dot(dx, dx) * (window - 2))
        direct_p = ols_slope(x, y)[2]
        for alpha in [0.2, direct_p, math.nextafter(direct_p, 0), math.nextafter(direct_p, 1)]:
            actual = flat_slope_windows(x, y, window, alpha, np.ones(1, dtype=bool))
            assert bool(actual[0]) == (direct_p >= alpha)


def test_fit_scaling_does_not_overflow_or_erase_small_residuals() -> None:
    x = np.array([-1e308, -5e307, 0.0, 5e307, 1e308])
    with np.errstate(all="raise"):
        assert ols_slope(x, x) == (1.0, 0.0, 0.0)
        assert ols_slope(np.arange(30) * 1e-300, np.ones(30)) == (0.0, 0.0, 1.0)
    x = np.arange(30, dtype=float)
    y = np.sin(x) + .03 * x
    original = ols_slope(x, y)
    scaled = ols_slope(x * 1e-150, y * 1e150)
    assert scaled[0] / 1e300 == pytest.approx(original[0])
    assert scaled[1] / 1e300 == pytest.approx(original[1])
    assert scaled[2] == pytest.approx(original[2])


@pytest.mark.parametrize("alpha", [-1, 0, 1e-300, 0.001, 0.2, 0.8, 1, 2, math.nan])
def test_gaps_constants_step_jumps_and_candidate_masks(alpha: float) -> None:
    rng = np.random.default_rng(117)
    x = np.cumsum(rng.choice([1, 3, 10000], 1600)).astype(float)
    x[600:] += 1e14
    y = np.r_[
        rng.normal(5, 0.001, 400),
        np.full(400, 5.0),
        rng.normal(5, 1e-6, 400),
        rng.normal(5, 0.01, 400),
    ]
    y[19], y[378], y[901] = math.nan, math.inf, -math.inf
    x[1200:1300] = x[1200]
    window = 30
    candidates = rng.random(len(y) - window + 1) < 0.7
    expected = []
    for i, candidate in enumerate(candidates):
        a, b = x[i : i + window], y[i : i + window]
        expected.append(
            candidate
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and ols_slope(a, b)[2] >= alpha
        )
    with np.errstate(all="raise"):
        actual = flat_slope_windows(x, y, window, alpha, candidates)
    np.testing.assert_array_equal(actual, expected)


def test_fast_path_really_avoids_one_fit_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import trainspotter.detectors._rolling_slope as rolling

    calls = 0
    direct = rolling.ols_slope

    def counted(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        nonlocal calls
        calls += 1
        return direct(x, y)

    monkeypatch.setattr(rolling, "ols_slope", counted)
    x = np.arange(100_000, dtype=float)
    y = 1 + 1 / (x + 10)
    mask = flat_slope_windows(x, y, 10_000, 0.2, np.ones(90_001, dtype=bool))
    assert not mask.any()  # every window is monotonically improving
    assert calls < 100  # accuracy fallbacks cannot silently restore quadratic work
