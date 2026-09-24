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

**Convergence suppression.** A flat window is *not* reported if either:
- it has already recovered `convergence_frac` (default 80%) of the metric's
  total observed drop (`first value - running minimum`) -- i.e. the curve
  is flat near its own floor, not stalled partway down; or
- an `lr` metric is present and has decayed to `lr_decay_frac` (default
  30%) or less of its peak by the window's start -- a schedule that has
  mostly finished decaying is expected to produce a flat loss.

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

import numpy as np

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

            start_value = float(values[0])
            total_drop = start_value - float(np.min(values))
            lr_steps, lr_values, peak_lr = self._lr_context(run)

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
                if self._looks_converged(
                    window_level=float(np.median(seg)),
                    start_value=start_value,
                    total_drop=total_drop,
                    window_start_step=float(steps[lo]),
                    lr_steps=lr_steps,
                    lr_values=lr_values,
                    peak_lr=peak_lr,
                ):
                    continue  # flat because it converged, not because it stalled
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
                            "Check for a frozen/detached parameter group if the plateau starts abruptly.",
                        ],
                    )
                )
        return findings

    @staticmethod
    def _lr_context(run: Run) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        lr_series = run.get("lr")
        if lr_series is None or len(lr_series) < 2:
            return None, None, 0.0
        lr_steps, lr_values = series_arrays(lr_series)
        return lr_steps, lr_values, float(np.max(lr_values))

    def _looks_converged(
        self,
        *,
        window_level: float,
        start_value: float,
        total_drop: float,
        window_start_step: float,
        lr_steps: np.ndarray | None,
        lr_values: np.ndarray | None,
        peak_lr: float,
    ) -> bool:
        if total_drop > 0:
            achieved = (start_value - window_level) / total_drop
            if achieved >= self.convergence_frac:
                return True
        if lr_steps is not None and lr_values is not None and peak_lr > 0:
            idx = int(np.argmin(np.abs(lr_steps - window_start_step)))
            if float(lr_values[idx]) <= self.lr_decay_frac * peak_lr:
                return True
        return False
