"""Shared types and numeric primitives every detector builds on."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from trainspotter.model import MetricSeries, Run

Severity = Literal["info", "warning", "error"]

_SEVERITY_RANK: dict[Severity, int] = {"info": 0, "warning": 1, "error": 2}


def severity_at_least(sev: Severity, floor: Severity) -> bool:
    return _SEVERITY_RANK[sev] >= _SEVERITY_RANK[floor]


@dataclass(slots=True)
class Finding:
    """One diagnosed pathology: what, where, how sure, and what to do."""

    detector: str
    title: str
    severity: Severity
    metric: str
    step_start: int
    step_end: int
    message: str
    evidence: dict[str, float | int | str] = field(default_factory=dict)
    fixes: list[str] = field(default_factory=list)


class Detector:
    """Base class for a detector. Subclasses implement `run`; the docstring
    of each subclass IS its documented algorithm -- surfaced verbatim by
    `trainspotter explain` and the README detector table."""

    name: str = "base"

    def run(self, run: Run) -> list[Finding]:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Numeric primitives shared across detectors.
# ---------------------------------------------------------------------------


def rolling_median_mad(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing rolling median and MAD (median absolute deviation) at each
    index, using only values up to and including that index (so it works
    identically in a post-hoc pass and in `watch` streaming mode). The first
    `window - 1` points use whatever history is available, shrinking the
    effective window rather than producing NaN."""
    n = len(values)
    med = np.empty(n, dtype=float)
    mad = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        m = float(np.median(chunk))
        med[i] = m
        mad[i] = float(np.median(np.abs(chunk - m)))
    return med, mad


def robust_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Iglewicz & Hoaglin's modified z-score: 0.6745 * (x - median) / MAD,
    using a trailing rolling median/MAD (see `rolling_median_mad`). The
    0.6745 constant makes MAD a consistent estimator of the standard
    deviation for normally distributed data, so this z is comparable in
    scale to an ordinary z-score. A MAD of exactly 0 (a flat run of
    identical values) is floored to avoid division by zero, which would
    otherwise manufacture infinite z-scores from noiseless data."""
    med, mad = rolling_median_mad(values, window)
    mad_floor = np.maximum(mad, 1e-12)
    result: np.ndarray = 0.6745 * (values - med) / mad_floor
    return result


def group_consecutive(indices: list[int], gap: int = 1) -> list[list[int]]:
    """Split a sorted list of indices into runs where consecutive members
    differ by at most `gap`, so e.g. a spike that trips the threshold for
    three steps in a row is reported as one finding, not three."""
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx - groups[-1][-1] <= gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least squares slope of y on x, its standard error, and the
    two-sided p-value against H0: slope == 0, via a normal approximation to
    the t-distribution (accurate for n >= ~30; for smaller n this is
    slightly anti-conservative, which detectors account for by also
    requiring a minimum window size before trusting the test)."""
    n = len(x)
    if n < 3:
        return 0.0, float("inf"), 1.0
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx == 0.0:
        return 0.0, float("inf"), 1.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    residuals = y - (slope * x + intercept)
    dof = n - 2
    if dof <= 0:
        return slope, float("inf"), 1.0
    resid_var = float(np.sum(residuals**2) / dof)
    se = math.sqrt(resid_var / sxx) if sxx > 0 else float("inf")
    if se == 0.0:
        return slope, 0.0, 0.0
    t_stat = slope / se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return slope, se, p_value


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def series_arrays(series: MetricSeries) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array(series.steps(), dtype=float)
    values = np.array(series.values(), dtype=float)
    return steps, values
