"""Plateau detector.

**Algorithm.** Slide a window of `window` points (default: 10% of the run,
floored at `min_window`=30) over the metric. In each window, fit an OLS
line and test H0: slope == 0 via the normal-approximation t-test in
`detectors.base.ols_slope`. A window is "flat" if p >= `alpha` (default
0.2 -- deliberately loose: we want to fail to reject "no trend", so a
*higher* p threshold flags more windows as flat, the opposite of a
significance test's usual use) AND the window's total relative change
`|last - first| / (|first| + eps)` is below `rel_change` (default 2%), so
a slope that's statistically insignificant only because the window is
short doesn't get flagged as a plateau. Flagged windows within one
`window`-length of each other are merged into a single finding (not just
strictly-consecutive ones): a near-zero metric can have enough relative
jitter that isolated windows in the middle of an obviously flat stretch
fail the `rel_change` guard by chance, which without this would fragment
one flat region into a dozen near-duplicate findings -- caught by running
this on the overfitting example, where train/loss flattens near zero for
hundreds of steps.

**False-positive modes.**
- A metric that's already near its achievable floor (e.g. eval loss close
  to the entropy of the label noise) will always look "flat" -- that's
  actually convergence, not a stall. This detector can't tell the
  difference from the curve alone; cross-check against expected loss
  floors for the task.
- Very noisy metrics can fail the significance test (p large) purely from
  variance, not from a true zero slope -- the `rel_change` guard mitigates
  but doesn't eliminate this.
- Small windows (short runs) make the significance test's normal
  approximation less accurate; findings on runs under ~2*min_window steps
  should be read as indicative, not definitive.
"""

from __future__ import annotations

from trainspotter.model import Run

from .base import Detector, Finding, group_consecutive, ols_slope, series_arrays

DEFAULT_METRICS = ("train/loss",)


class PlateauDetector(Detector):
    name = "plateau"

    def __init__(
        self,
        metrics: tuple[str, ...] = DEFAULT_METRICS,
        min_window: int = 30,
        window_frac: float = 0.10,
        alpha: float = 0.2,
        rel_change: float = 0.02,
    ) -> None:
        self.metrics = metrics
        self.min_window = min_window
        self.window_frac = window_frac
        self.alpha = alpha
        self.rel_change = rel_change

    def run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None:
                continue
            steps, values = series_arrays(series)
            n = len(values)
            window = max(self.min_window, int(n * self.window_frac))
            if n < window + 1:
                continue
            flat_windows: list[int] = []
            for start in range(0, n - window + 1):
                w_steps = steps[start : start + window]
                w_values = values[start : start + window]
                _slope, _se, p_value = ols_slope(w_steps, w_values)
                first, last = w_values[0], w_values[-1]
                rel = abs(last - first) / (abs(first) + 1e-12)
                if p_value >= self.alpha and rel < self.rel_change:
                    flat_windows.append(start)
            for group in group_consecutive(flat_windows, gap=window):
                lo, hi = group[0], group[-1] + window - 1
                seg = values[lo : hi + 1]
                findings.append(
                    Finding(
                        detector=self.name,
                        title="Plateau",
                        severity="info",
                        metric=metric_name,
                        step_start=int(steps[lo]),
                        step_end=int(steps[hi]),
                        message=(
                            f"{metric_name} barely moved from step {int(steps[lo])} to "
                            f"{int(steps[hi])} ({hi - lo + 1} points, range "
                            f"{float(seg.min()):.4g}-{float(seg.max()):.4g})."
                        ),
                        evidence={
                            "window": window,
                            "alpha": self.alpha,
                            "rel_change_threshold": self.rel_change,
                            "value_range": float(seg.max() - seg.min()),
                        },
                        fixes=[
                            "If this is early training, check warmup finished and LR reached its target.",
                            "Try a higher LR, an LR-range test, or a different schedule (cosine/one-cycle) if the plateau is unexpected.",
                            "If it's late in training near a plausible floor, this may just be convergence -- not a problem.",
                            "Check for a frozen/detached parameter group if the plateau starts abruptly.",
                        ],
                    )
                )
        return findings
