"""Eval-noise detector: is this metric too noisy to rank checkpoints by?

**Algorithm.** Let `d` be the first differences of the eval metric (values
in logged order). Compute `step_noise = median(|d|)` (a robust estimate of
typical step-to-step jitter) and `total_improvement = |best - worst|` over
the whole run. The noise ratio is `step_noise / (total_improvement + eps)`.
If that ratio exceeds `ratio_threshold` (default 0.15) -- i.e. a single
step's typical jitter is at least 15% of the *entire run's* improvement --
picking the single best-looking checkpoint by this metric alone is
unreliable, because two checkpoints could differ by less than the metric's
own noise floor.

**Why this matters, concretely.** If checkpoint selection is "take the
step with the lowest eval loss", and adjacent checkpoints differ by less
than typical noise, that selection is effectively random among the top few
candidates.

**False-positive modes.**
- A run that's genuinely almost flat (e.g. already converged) has small
  `total_improvement` by construction, which inflates the ratio even if
  the metric isn't especially noisy in absolute terms -- read this
  alongside the plateau detector.
- A tiny eval set produces high variance by sampling error, not a flaw in
  the metric itself; the fix (bigger eval set / more eval frequency
  averaging) is the same either way.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, series_arrays

DEFAULT_RATIO_THRESHOLD = 0.15


class EvalNoiseDetector(Detector):
    name = "eval_noise"

    def __init__(self, metric: str = "eval/loss", ratio_threshold: float = DEFAULT_RATIO_THRESHOLD, min_points: int = 6) -> None:
        self.metric = metric
        self.ratio_threshold = ratio_threshold
        self.min_points = min_points

    def run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < self.min_points:
            return []
        steps, values = series_arrays(series)
        diffs = np.diff(values)
        if len(diffs) == 0:
            return []
        step_noise = float(np.median(np.abs(diffs)))
        total_improvement = float(np.max(values) - np.min(values))
        ratio = step_noise / (total_improvement + 1e-12)
        if ratio < self.ratio_threshold:
            return []
        return [
            Finding(
                detector=self.name,
                title="Eval metric too noisy to rank checkpoints",
                severity="warning",
                metric=self.metric,
                step_start=int(steps[0]),
                step_end=int(steps[-1]),
                message=(
                    f"Typical step-to-step jitter in {self.metric} ({step_noise:.4g}) is "
                    f"{ratio * 100:.0f}% of the whole run's total improvement "
                    f"({total_improvement:.4g}) -- adjacent checkpoints may not be "
                    "meaningfully different."
                ),
                evidence={"step_noise": step_noise, "total_improvement": total_improvement, "ratio": ratio},
                fixes=[
                    "Evaluate on more data, or average multiple eval passes/seeds before comparing checkpoints.",
                    "Select checkpoints by a smoothed (e.g. EMA) version of this metric instead of the raw value.",
                    "Evaluate less frequently but on a larger sample each time, if eval cost is the constraint.",
                ],
            )
        ]
