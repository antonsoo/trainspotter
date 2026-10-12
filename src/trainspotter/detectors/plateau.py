"""Plateau detector.

**Algorithm.** Within each finite segment, slide a window of `window` points
(default: 10% of the segment, floored at `min_window`=30). Retain the original
OLS non-significance (p >= `alpha`, default 0.2) and raw endpoint-change
(< `rel_change`, default 2%) filters, but do not treat either as evidence of
a stable level. A symmetric excursion can pass both.

Also bound the centered-median level's full range. Its smoothing window is
half the assessment window, rounded up to odd, minimum 3 and capped at 201
points. The range limit is the larger of `rel_change` times the largest
absolute level (floored at 1e-12), or an allowance for observation scatter.
The allowance is 6 * 1.2533 * 1.4826 * MAD(first differences) / sqrt(2 * m),
where m is the smoothing width. Difference MAD uses the last min(window-1,
201) differences in the assessment window. This avoids estimating noise
from a mixture of distinct constant levels. These normal-reference constants
are a heuristic scale, not a confidence bound for correlated training losses.

Windows may join only if their intervals overlap or touch AND their union
meets the level-range limit, using the smallest noise allowance among its
windows. Short noise need not fragment a plateau, but a chain of individually
flat windows cannot silently turn a substantial level change into one flat
region. The report calls this a possible plateau, distinguishes the smoothed
level from raw readings, and retains the noise allowance as numeric evidence.

**Convergence suppression.** A flat window is *not* reported if either:
- it has already recovered `convergence_frac` (default 80%) of the metric's
  segment's observed drop (`first value - segment minimum`) -- i.e. the curve
  is flat near its own floor, not stalled partway down; or
- an `lr` metric is present and has decayed to `lr_decay_frac` (default
  30%) or less of its observed peak by the window's start -- a schedule that has
  mostly finished decaying is expected to produce a flat loss.

The LR context uses only readings at or before the finding's start. A NaN,
Inf or negative LR resets that context, and a future peak cannot explain an
earlier loss. Loss gaps likewise cannot poison or connect segment baselines.

Ordinary end-of-training convergence on a cosine/warmup schedule -- the
common case -- trips both, so it's suppressed entirely rather than
reported as a finding: caught on `examples/baseline`, an unremarkable
healthy run, which produced 4 "Plateau" infos before this existed. Both
checks assume lower-is-better semantics (a loss, not an accuracy); point
this detector only at metrics where "flat" can mean "stalled" (its
default, `train/loss`).

**False-positive modes.**
- A real stall that happens to sit near a coincidentally low value, or
  that starts right as an unrelated LR decay is finishing, can be wrongly
  suppressed by the convergence checks above -- they're a heuristic
  reading of the curve's *shape*, not a check that training actually
  succeeded.
- Very noisy metrics can fail the significance test (p large) purely from
  variance, not from a true zero slope -- the `rel_change` guard mitigates
  but doesn't eliminate this.
- Small windows (short runs) make the significance test's normal
  approximation less accurate; findings on runs under ~2*min_window steps
  should be read as indicative, not definitive.
"""

from __future__ import annotations

from collections import deque
from math import sqrt

import numpy as np

from trainspotter.model import Run

from ._rolling_slope import flat_slope_windows
from .base import (
    MAX_LEVEL_WINDOW,
    Detector,
    Finding,
    FloatArray,
    add_segment_evidence,
    centered_median,
    finite_spans,
    rolling_median_mad,
    series_arrays,
)

DEFAULT_METRICS = ("train/loss",)


def _window_extrema(values: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Exact full-window min/max in O(n) work and O(window) queue space."""
    minima = np.empty(len(values) - window + 1, dtype=float)
    maxima = np.empty_like(minima)
    low: deque[int] = deque()
    high: deque[int] = deque()
    for i, value in enumerate(values):
        while low and low[0] <= i - window:
            low.popleft()
        while high and high[0] <= i - window:
            high.popleft()
        while low and values[low[-1]] >= value:
            low.pop()
        while high and values[high[-1]] <= value:
            high.pop()
        low.append(i)
        high.append(i)
        if i >= window - 1:
            minima[i - window + 1] = values[low[0]]
            maxima[i - window + 1] = values[high[0]]
    return minima, maxima


def _relative_range(low: float, high: float) -> float:
    scale = max(abs(low), abs(high), 1e-12)
    return high / scale - low / scale


def _stable_groups(
    starts: list[int],
    minima: FloatArray,
    maxima: FloatArray,
    window: int,
    tolerance: float,
    noise_allowance: FloatArray,
) -> list[tuple[int, int, float, float, int, float]]:
    """Join overlapping windows only while their combined level remains stable."""
    groups: list[tuple[int, int, float, float, int, float]] = []
    for start in starts:
        low, high = float(minima[start]), float(maxima[start])
        allowance = float(noise_allowance[start])
        end = start + window - 1
        if groups:
            first, previous_end, old_low, old_high, count, old_allowance = groups[-1]
            merged_low, merged_high = min(old_low, low), max(old_high, high)
            combined_allowance = min(old_allowance, allowance)
            scale = max(abs(merged_low), abs(merged_high), 1e-12)
            bound = max(tolerance, combined_allowance / scale)
            if start <= previous_end + 1 and _relative_range(merged_low, merged_high) < bound:
                groups[-1] = first, end, merged_low, merged_high, count + 1, combined_allowance
                continue
        groups.append((start, end, low, high, 1, allowance))
    return groups


class PlateauDetector(Detector):
    name = "plateau"

    def __init__(
        self,
        metrics: tuple[str, ...] = DEFAULT_METRICS,
        min_window: int = 30,
        window_frac: float = 0.10,
        alpha: float = 0.2,
        rel_change: float = 0.02,
        convergence_frac: float = 0.8,
        lr_decay_frac: float = 0.3,
    ) -> None:
        self.metrics = metrics
        self.min_window = min_window
        self.window_frac = window_frac
        self.alpha = alpha
        self.rel_change = rel_change
        self.convergence_frac = convergence_frac
        self.lr_decay_frac = lr_decay_frac

    def _run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        lr_steps, lr_values, lr_peaks = self._lr_context(run)
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None:
                continue
            steps, values = series_arrays(series)
            spans = finite_spans(values)
            has_unreadable = series.readable_count() != len(series)
            partial = len(spans) != 1 or spans[0] != (0, len(values))
            for start, stop in spans:
                segment = self._segment(
                    metric_name,
                    steps[start:stop],
                    values[start:stop],
                    lr_steps,
                    lr_values,
                    lr_peaks,
                )
                if partial:
                    add_segment_evidence(segment, steps[start:stop],
                                         unreadable=has_unreadable)
                findings.extend(segment)
        return findings

    def _segment(
        self,
        metric_name: str,
        steps: FloatArray,
        values: FloatArray,
        lr_steps: FloatArray | None,
        lr_values: FloatArray | None,
        lr_peaks: FloatArray | None,
    ) -> list[Finding]:
        n = len(values)
        window = max(3, self.min_window, int(n * self.window_frac))
        if n < window:
            return []
        first, last = values[: n - window + 1], values[window - 1 :]
        with np.errstate(invalid="ignore", over="ignore"):
            rel = np.abs(last - first) / (np.abs(first) + 1e-12)
        endpoint_candidates = rel < self.rel_change
        if not np.any(endpoint_candidates):
            return []
        start_value = float(values[0])
        total_drop = start_value - float(np.min(values))

        # A nonsignificant linear slope and matching endpoints can conceal a
        # large symmetric rise. Bound the actual median level as well, including
        # across the union of windows; each condition measures a different fact.
        smoothing_window = min(MAX_LEVEL_WINDOW, max(3, (window // 2) | 1))
        constant = bool(np.all(values == values[0]))
        minima: FloatArray
        maxima: FloatArray
        if constant:
            # Preserve the exact window result without sorting millions of
            # identical observations or grouping every identical window.
            minima = maxima = np.full(n - window + 1, values[0])
        else:
            smooth = centered_median(values, smoothing_window)
            minima, maxima = _window_extrema(smooth, window)
        scale = np.maximum(np.maximum(np.abs(minima), np.abs(maxima)), 1e-12)
        # Estimate observation scatter from differences, not from the mixture
        # of loss levels: isolated level jumps must not inflate the allowance.
        # Cap this window too, so long logs do not restore quadratic work.
        noise_window = min(MAX_LEVEL_WINDOW, max(1, window - 1))
        noise_allowance: FloatArray
        if constant:
            noise_allowance = np.zeros(n - window + 1)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                _, difference_mad = rolling_median_mad(np.diff(values), noise_window)
                noise_allowance = (
                    6 * 1.2533 * 1.4826 / sqrt(2 * smoothing_window) * difference_mad[window - 2 :]
                )
        stable = np.isfinite(noise_allowance) & (
            maxima / scale - minima / scale
            < np.maximum(
                self.rel_change,
                noise_allowance / scale,
            )
        )
        flat = flat_slope_windows(
            steps,
            values,
            window,
            self.alpha,
            stable & endpoint_candidates,
        )
        if constant:
            level = float(values[0])
            groups = (
                [(0, n - 1, level, level, int(np.count_nonzero(flat)), 0.0)] if np.any(flat) else []
            )
        else:
            groups = _stable_groups(
                np.flatnonzero(flat).tolist(),
                minima,
                maxima,
                window,
                self.rel_change,
                noise_allowance,
            )
        findings = []
        for lo, hi, level_min, level_max, count, allowance in groups:
            seg = values[lo : hi + 1]
            if self._looks_converged(
                window_level=float(np.median(seg)),
                start_value=start_value,
                total_drop=total_drop,
                window_start_step=float(steps[lo]),
                lr_steps=lr_steps,
                lr_values=lr_values,
                lr_peaks=lr_peaks,
            ):
                continue
            findings.append(
                Finding(
                    detector=self.name,
                    title="Possible plateau",
                    severity="info",
                    metric=metric_name,
                    step_start=int(steps[lo]),
                    step_end=int(steps[hi]),
                    message=(
                        f"The median-smoothed {metric_name} level stayed between "
                        f"{level_min:.4g} and {level_max:.4g} from step {int(steps[lo])} to "
                        f"{int(steps[hi])} ({hi - lo + 1} points; raw readings range "
                        f"{float(seg.min()):.4g}-{float(seg.max()):.4g})."
                    ),
                    evidence={
                        "window": window,
                        "alpha": self.alpha,
                        "rel_change_threshold": self.rel_change,
                        "value_range": float(seg.max() - seg.min()),
                        "smoothing_window": smoothing_window,
                        "level_min": level_min,
                        "level_max": level_max,
                        "relative_level_range": _relative_range(level_min, level_max),
                        "noise_allowance": allowance,
                        "noise_window": noise_window,
                        "qualifying_windows": count,
                    },
                    fixes=[
                        "If this is early training, check warmup finished and LR reached its target.",
                        "Try a higher LR, an LR-range test, or a different schedule (cosine/one-cycle) if the plateau is unexpected.",
                        "Check for a frozen/detached parameter group if the plateau starts abruptly.",
                    ],
                )
            )
        return findings

    @staticmethod
    def _lr_context(run: Run) -> tuple[FloatArray | None, FloatArray | None, FloatArray | None]:
        lr_series = run.get("lr")
        if lr_series is None or len(lr_series) < 2:
            return None, None, None
        lr_steps, lr_values = series_arrays(lr_series)
        peaks = np.full(len(lr_values), np.nan)
        valid = np.where(lr_values >= 0, lr_values, np.nan)
        for start, stop in finite_spans(valid):
            peaks[start:stop] = np.maximum.accumulate(lr_values[start:stop])
        return lr_steps, lr_values, peaks

    def _looks_converged(
        self,
        *,
        window_level: float,
        start_value: float,
        total_drop: float,
        window_start_step: float,
        lr_steps: FloatArray | None,
        lr_values: FloatArray | None,
        lr_peaks: FloatArray | None,
    ) -> bool:
        if total_drop > 0:
            achieved = (start_value - window_level) / total_drop
            if achieved >= self.convergence_frac:
                return True
        if lr_steps is not None and lr_values is not None and lr_peaks is not None:
            # Only a reading at or before the finding can explain its schedule.
            idx = int(np.searchsorted(lr_steps, window_start_step, side="right")) - 1
            if (
                idx >= 0
                and lr_peaks[idx] > 0
                and lr_values[idx] <= self.lr_decay_frac * lr_peaks[idx]
            ):
                return True
        return False
