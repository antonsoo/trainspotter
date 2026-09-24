"""Overfitting-onset detector.

**Algorithm.** Requires both a train and an eval metric (default
`train/loss`, `eval/loss`). Find the step `k*` of the eval metric's running
minimum (the best checkpoint so far at every point). After `k*`, look at
the remaining eval points: fit an OLS slope (via `detectors.base.ols_slope`)
and require it to be positive with p < `alpha` (default 0.1 -- looser than
the divergence test, because overfitting onset is a real, common event we
want to catch early rather than miss). Also require at least
`min_points_after` (default 3) eval points after `k*`, and that eval has
risen at least `min_rise` (default 1%) above its minimum, so a single noisy
uptick right at the end doesn't fire. The train metric isn't required to
be strictly monotonic -- only that it hasn't also turned upward at the same
point (which would indicate divergence, not overfitting: see the
`divergence` detector).

**False-positive modes.**
- A noisy eval set (small validation split) can show a spurious upward
  eval trend after the true best point by chance; see the `eval_noise`
  detector, which flags exactly this risk and should be read alongside
  this one.
- A change in eval *data* mid-run (curriculum, harder eval subset added
  later) looks identical to genuine overfitting.
- LR schedule changes near `k*` (e.g. decay finishing) can cause both
  metrics to move in ways that coincidentally resemble this pattern.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, ols_slope, series_arrays


class OverfittingDetector(Detector):
    name = "overfitting"

    def __init__(
        self,
        train_metric: str = "train/loss",
        eval_metric: str = "eval/loss",
        alpha: float = 0.1,
        min_points_after: int = 3,
        min_rise: float = 0.01,
    ) -> None:
        self.train_metric = train_metric
        self.eval_metric = eval_metric
        self.alpha = alpha
        self.min_points_after = min_points_after
        self.min_rise = min_rise

    def run(self, run: Run) -> list[Finding]:
        eval_series = run.get(self.eval_metric)
        if eval_series is None or len(eval_series) < self.min_points_after + 1:
            return []
        eval_steps, eval_values = series_arrays(eval_series)
        best_i = int(np.argmin(eval_values))
        after = eval_values[best_i:]
        after_steps = eval_steps[best_i:]
        if len(after) - 1 < self.min_points_after:
            return []

        slope, _se, p_value = ols_slope(after_steps, after)
        best_value = float(eval_values[best_i])
        current_value = float(eval_values[-1])
        rise = (current_value - best_value) / (abs(best_value) + 1e-12)

        if not (slope > 0 and p_value < self.alpha and rise >= self.min_rise):
            return []

        train_series = run.get(self.train_metric)
        train_note = ""
        if train_series is not None and len(train_series) >= 2:
            t_steps, t_values = series_arrays(train_series)
            mask = t_steps >= eval_steps[best_i]
            if mask.sum() >= 2:
                t_slope, _t_se, t_p = ols_slope(t_steps[mask], t_values[mask])
                if t_slope < 0 and t_p < 0.2:
                    train_note = (
                        f" Meanwhile {self.train_metric} kept falling over the same range "
                        "(slope < 0) -- the classic overfitting signature."
                    )

        return [
            Finding(
                detector=self.name,
                title="Overfitting onset",
                severity="warning",
                metric=self.eval_metric,
                step_start=int(eval_steps[best_i]),
                step_end=int(eval_steps[-1]),
                message=(
                    f"Best {self.eval_metric} was {best_value:.4g} at step "
                    f"{int(eval_steps[best_i])}; it's since risen to {current_value:.4g} "
                    f"({rise * 100:.1f}%, slope p={p_value:.3g})."
                )
                + train_note,
                evidence={
                    "best_step": int(eval_steps[best_i]),
                    "best_value": best_value,
                    "current_value": current_value,
                    "relative_rise": rise,
                    "slope_p_value": p_value,
                },
                fixes=[
                    f"Use the checkpoint at step {int(eval_steps[best_i])} (best {self.eval_metric}), not the last one.",
                    "Add or increase regularization: weight decay, dropout, data augmentation.",
                    "Reduce model capacity or train for fewer steps / add early stopping.",
                    "Get more training data, or de-duplicate against the eval set if overlap is possible.",
                ],
            )
        ]
