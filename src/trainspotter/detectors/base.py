"""Shared types and numeric primitives every detector builds on."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from trainspotter.model import MetricSeries, Run

Severity = Literal["info", "warning", "error"]

#: Steps and values as the detectors hold them. Spelled out because a bare `np.ndarray` is
#: only a complete type on the numpy releases that give its parameters defaults, and those
#: releases don't install on Python 3.10.
FloatArray = npt.NDArray[np.float64]

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
    """Base class for a detector. Subclasses implement `_run`; the docstring
    of each subclass IS its documented algorithm -- surfaced verbatim by
    the README detector table."""

    name: str = "base"

    def run(self, run: Run) -> list[Finding]:
        """Evaluate unique step observations, preserving ambiguous input in reports."""
        return self._run(run.analysis_view())

    def _run(self, run: Run) -> list[Finding]:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Numeric primitives shared across detectors.
# ---------------------------------------------------------------------------


def _window_batches(values: FloatArray, window: int) -> Iterator[tuple[int, FloatArray]]:
    """Bound the copies made by median/partition, not just the strided view.

    A view of every 201-point window is cheap, but reducing that entire view
    would allocate about 1.6 GB for a million readings. Each reduction here
    handles at most 262,144 doubles (2 MiB), or one unusually wide window.
    The views stay read-only: partition must never rearrange the input history.
    """
    views = np.lib.stride_tricks.sliding_window_view(values, window)
    batch = max(1, 262_144 // window)
    for lo in range(0, len(views), batch):
        yield lo, views[lo : lo + batch]


def trailing_median(values: FloatArray, window: int) -> FloatArray:
    """Median including the current reading, with shrinking initial windows."""
    if window < 1:
        raise ValueError("window must be positive")
    med = np.empty(len(values), dtype=float)
    if not len(values):
        return med
    window = min(window, len(values))
    for i in range(window - 1):
        med[i] = np.median(values[: i + 1])
    for lo, views in _window_batches(values, window):
        med[lo + window - 1 : lo + window - 1 + len(views)] = np.median(views, axis=1)
    return med


def rolling_median_mad(values: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Trailing rolling median and MAD (median absolute deviation) at each
    index, using only values up to and including that index (so it works
    identically in a post-hoc pass and in `watch` streaming mode). The first
    `window - 1` points use whatever history is available, shrinking the
    effective window rather than producing NaN."""
    if window < 1:
        raise ValueError("window must be positive")
    n = len(values)
    med = np.empty(n, dtype=float)
    mad = np.empty(n, dtype=float)
    if not n:
        return med, mad
    window = min(window, n)
    for i in range(window - 1):
        chunk = values[: i + 1]
        m = float(np.median(chunk))
        med[i] = m
        mad[i] = float(np.median(np.abs(chunk - m)))
    for lo, views in _window_batches(values, window):
        dest = slice(lo + window - 1, lo + window - 1 + len(views))
        med[dest] = np.median(views, axis=1)
        deviations = np.abs(views - med[dest, None])
        mad[dest] = np.median(deviations, axis=1, overwrite_input=True)
    return med, mad


def robust_zscore(values: FloatArray, window: int) -> FloatArray:
    """Iglewicz & Hoaglin's modified z-score: 0.6745 * (x - median) / MAD,
    using a trailing rolling median/MAD (see `rolling_median_mad`). The
    0.6745 constant makes MAD a consistent estimator of the standard
    deviation for normally distributed data, so this z is comparable in
    scale to an ordinary z-score. A MAD of exactly 0 (a flat run of
    identical values) is floored to avoid division by zero, which would
    otherwise manufacture infinite z-scores from noiseless data."""
    med, mad = rolling_median_mad(values, window)
    mad_floor = np.maximum(mad, 1e-12)
    result: FloatArray = 0.6745 * (values - med) / mad_floor
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


def finite_spans(values: FloatArray) -> list[tuple[int, int]]:
    """Half-open contiguous spans; invalid readings are gaps, not observations."""
    boundaries = np.flatnonzero(np.diff(np.r_[False, np.isfinite(values), False]))
    return [(int(start), int(stop)) for start, stop in boundaries.reshape(-1, 2)]


def ols_slope(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """Ordinary least squares slope of y on x, its standard error, and the
    two-sided p-value against H0: slope == 0, via a normal approximation to
    the t-distribution (accurate for n >= ~30; for smaller n this is
    slightly anti-conservative, which detectors account for by also
    requiring a minimum window size before trusting the test).

    A window that contains a NaN/Inf reading (a diverged run legitimately
    produces these) can't support a slope estimate; rather than let numpy
    warn its way through arithmetic on non-finite values and return NaN
    results a caller has to know to distrust, that case is treated the
    same as "not enough data": no significant slope, p=1."""
    n = len(x)
    if n < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return 0.0, float("inf"), 1.0
    # Fit translated, scaled coordinates. Reconstructing y from a large
    # intercept loses small residuals, and squaring unscaled readings can
    # overflow even when both the slope and its uncertainty are representable.
    dx, x_scale, x_factor = _center_for_fit(x)
    dy, y_scale, y_factor = _center_for_fit(y)
    sxx = float(np.sum(dx**2))
    if sxx == 0.0:
        return 0.0, float("inf"), 1.0
    unit_slope = float(np.sum(dx * dy) / sxx)
    residuals = dy - unit_slope * dx
    unit_se = math.sqrt(float(np.sum(residuals**2)) / (n - 2) / sxx)
    slope = _fit_units(unit_slope, y_scale, x_scale, y_factor / x_factor)
    se = _fit_units(unit_se, y_scale, x_scale, y_factor / x_factor)
    if unit_se == 0.0:
        return slope, se, 1.0 if unit_slope == 0.0 else 0.0
    t_stat = unit_slope / unit_se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return slope, se, p_value


def _center_for_fit(values: FloatArray) -> tuple[FloatArray, float, float]:
    with np.errstate(over="ignore"):
        shifted = values - values[0]
    factor = 1.0
    if not np.all(np.isfinite(shifted)):
        # Opposite-sign finite endpoints can have an unrepresentable difference.
        shifted = values / 2 - values[0] / 2
        factor = 2.0
    scale = float(np.max(np.abs(shifted))) or 1.0
    shifted /= scale
    shifted -= shifted.mean()
    return shifted, scale, factor


def _fit_units(value: float, y_scale: float, x_scale: float, factor: float) -> float:
    # Combine exponents before rounding. y_scale/x_scale alone can overflow
    # even when multiplying by a small fitted coefficient would be finite.
    v, ve = math.frexp(value)
    y, ye = math.frexp(y_scale)
    x, xe = math.frexp(x_scale)
    try:
        return math.ldexp(v * y / x * factor, ve + ye - xe)
    except OverflowError:
        return math.copysign(math.inf, value)


def format_p(p_value: float) -> str:
    """``p=0.0245``, or ``p<0.0001`` rather than a literal ``p=0``: the normal approximation
    underflows to exactly 0 for a strong slope, and no finite sample supports p = 0."""
    return "p<0.0001" if p_value < 1e-4 else f"p={p_value:.3g}"


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Levels: where a loss curve started, the best it reached, and where it ended.
#
# A logged training loss is one mini-batch's average: single points scatter widely
# around a level that moves slowly. What a reader cares about is the level, so the
# detectors that judge "did this hurt the run" work on these, not on raw points.
# ---------------------------------------------------------------------------

#: Fewer points than this and there is no level to estimate.
MIN_LEVEL_POINTS = 10
#: The smoothing window never exceeds this many points, so a setback that lasts a
#: few hundred logged points in a very long run is still seen.
MAX_LEVEL_WINDOW = 201
#: A level this share of the run's progress above its earlier best is an excursion. A
#: smaller share would count the slow wiggles of a flat curve; at a quarter, what is left are
#: rises.
EXCURSION_SHARE = 0.25


@dataclass(slots=True)
class Levels:
    """A loss series reduced to the levels it moved between."""

    smooth: FloatArray  #: centred rolling median, same length as the series
    window: int
    #: Where the run began: the median of its first three points or of its first 5%,
    #: whichever is higher. Many runs lose most of their loss in the first few logged
    #: points, and a 5% window would start measuring after that drop.
    start: float
    best: float  #: lowest smoothed level
    best_index: int
    final: float  #: median of the last 5% of points
    noise: float  #: robust standard deviation of the smoothed series around the level
    #: The same, as a fraction of the level, when every value is positive: a mini-batch
    #: loss scatters in proportion to its level, so early points are noisier in absolute terms.
    relative_noise: float | None = None
    scatter: float = 0.0  #: robust standard deviation of single points around the level
    relative_scatter: float | None = None  #: the same on the log scale, for a positive series
    start_noise: float = 0.0  #: how far the estimate of `start` strays by chance

    def noise_at(self, level: float) -> float:
        """The level's noise where the loss is around `level`."""
        if self.relative_noise is None:
            return self.noise
        return max(self.noise, self.relative_noise * abs(level))

    @property
    def progress(self) -> float:
        """How far the level fell from its start to its best; the yardstick every
        "how much did this cost" judgement is measured against."""
        return self.start - self.best

    @property
    def has_progress(self) -> bool:
        """Whether the level clearly fell. A flat, noisy series has a "best" a couple of
        noises below its start by chance alone, so the bar is six of the level's noise,
        and four of the noise in the start itself."""
        return self.progress > max(6.0 * self.noise, 4.0 * self.start_noise)

    @property
    def gave_back(self) -> float:
        """Share of the run's progress the final level sits above the best level."""
        return (self.final - self.best) / self.progress if self.progress > 0 else 0.0


def centered_median(values: FloatArray, window: int) -> FloatArray:
    """Rolling median over a window centred on each point (shrinking at the ends)."""
    if window < 1:
        raise ValueError("window must be positive")
    half = window // 2
    n = len(values)
    med = np.empty(n, dtype=float)
    width = 2 * half + 1  # Preserve the symmetric convention for even window arguments.
    if width > n:
        return np.array([np.median(values[max(0, i - half) : i + half + 1]) for i in range(n)], dtype=float)
    for i in range(half):
        med[i] = np.median(values[: i + half + 1])
    for lo, views in _window_batches(values, width):
        med[lo + half : lo + half + len(views)] = np.median(views, axis=1)
    for i in range(n - half, n):
        med[i] = np.median(values[i - half :])
    return med


def levels(values: FloatArray) -> Levels | None:
    """Estimate one finite segment's levels; callers must split at NaN/Inf.

    Filtering here would erase the gap and let a rolling window, baseline or
    recovery claim borrow observations from the other side of it.
    """
    if not np.all(np.isfinite(values)):
        return None
    finite = values
    n = len(finite)
    if n < MIN_LEVEL_POINTS:
        return None
    edge = max(3, n // 20)
    window = min(max(5, n // 25), MAX_LEVEL_WINDOW) | 1
    smooth = centered_median(finite, window)
    best_index = int(np.argmin(smooth))
    residual = finite - smooth
    point_sd = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    # In a short series the window is five points wide and each residual is taken from a
    # median that includes the point itself, which understates the scatter. Successive
    # differences do not have that problem (and a trend shifts them all alike).
    change = np.diff(finite)
    point_sd = max(point_sd, 1.4826 * float(np.median(np.abs(change - np.median(change)))) / math.sqrt(2))
    # A median of w points has about 1.2533 / sqrt(w) of one point's standard deviation.
    shrink = 1.2533 / math.sqrt(window)
    noise = shrink * point_sd
    relative_scatter = None
    if float(finite.min()) > 0:
        ratio = np.log(finite / smooth)
        relative_scatter = 1.4826 * float(np.median(np.abs(ratio - np.median(ratio))))
    first_points, first_share = float(np.median(finite[:3])), float(np.median(finite[:edge]))
    start, start_count = (first_points, 3) if first_points > first_share else (first_share, edge)
    start_scatter = point_sd if relative_scatter is None else max(point_sd, relative_scatter * abs(start))
    return Levels(
        smooth=smooth,
        window=window,
        start=start,
        start_noise=1.2533 * start_scatter / math.sqrt(start_count),
        best=float(smooth[best_index]),
        best_index=best_index,
        final=float(np.median(finite[-edge:])),
        noise=noise,
        relative_noise=None if relative_scatter is None else shrink * relative_scatter,
        scatter=point_sd,
        relative_scatter=relative_scatter,
    )


@dataclass(slots=True)
class Excursion:
    """A stretch where the smoothed loss sat clearly above a level it had already reached."""

    start_index: int
    end_index: int
    base: float  #: the best level reached before it
    peak: float  #: the highest smoothed level during it
    recovered: bool  #: False when the log ends before the level comes back


def excursions(lv: Levels) -> list[Excursion]:
    """Find the stretches where the level rose well above its earlier best.

    "Well above" means by a quarter of the run's progress, and never less than
    four times the level's own noise at that point. An excursion ends when the level is
    back within a quarter of that margin of the earlier best.

    The yardstick is the run's progress rather than a ratio to the best level
    because loss differences, not ratios, are what carry meaning: a
    cross-entropy that goes from 0.10 to 0.15 has moved 0.05 nats, the same
    as one that goes from 2.00 to 2.05.
    """
    if not lv.has_progress:
        return []
    found: list[Excursion] = []
    running_best = lv.smooth[0]
    current: Excursion | None = None
    margin = 0.0
    for i, level in enumerate(lv.smooth):
        if current is None:
            running_best = min(running_best, level)
            margin = max(EXCURSION_SHARE * lv.progress, 4.0 * lv.noise_at(running_best))
            if level > running_best + margin:
                current = Excursion(i, i, float(running_best), float(level), recovered=False)
        else:
            current.end_index = i
            current.peak = max(current.peak, float(level))
            if level <= current.base + 0.25 * margin:
                current.recovered = True
                found.append(current)
                current = None
                running_best = min(running_best, level)
    if current is not None:
        found.append(current)
    return found


def progress_text(share: float, scope: str = "run") -> str:
    """``43% of the run's progress``, or ``29 times the run's progress`` once a
    percentage stops being readable."""
    return (
        f"{share:.0%} of the {scope}'s progress" if share < 10
        else f"{share:.0f} times the {scope}'s progress"
    )


def add_segment_evidence(
    findings: list[Finding], steps: FloatArray, *, unreadable: bool = False,
) -> None:
    """Name the finite observations behind a finding in a history with gaps."""
    for finding in findings:
        finding.evidence.update({
            "segment_start_step": int(steps[0]),
            "segment_end_step": int(steps[-1]),
            "segment_observations": len(steps),
        })
        finding.message += (
            f" Assessed within the finite segment at steps {int(steps[0])}-{int(steps[-1])} "
            f"({len(steps)} observations); no loss trajectory is inferred across "
            f"{'non-finite or unreadable values' if unreadable else 'NaN/Inf readings'}."
        )


def series_arrays(series: MetricSeries) -> tuple[FloatArray, FloatArray]:
    steps = np.array(series.steps(), dtype=float)
    values = np.array(series.values(), dtype=float)
    return steps, values
