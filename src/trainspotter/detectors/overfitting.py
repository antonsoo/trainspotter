"""Overfitting-onset detector.

**Algorithm.** Requires both a train and an eval metric (default
`train/loss`, `eval/loss`). Find the step `k*` of the eval metric's running
minimum (the best checkpoint so far at every point). After `k*`, look at
the remaining eval points: fit an OLS slope (via `detectors.base.ols_slope`)
and require it to be positive with p < `alpha` (default 0.1 -- looser than
the divergence test, because overfitting onset is a real, common event we
want to catch early rather than miss). Also require at least
`min_points_after` (default 3) eval points after `k*`, and that eval has
risen at least `min_rise` (default 3%) above its minimum.

**Severity follows the size of the rise.** Below `warn_rise` (default 10%)
the finding is `info`; from there it is a `warning`; from `error_rise`
(default 50%) an `error`: the last checkpoint is then far worse than the
best one. A rise of `clear_rise` (default 5%) is already a `warning` when
the trend is beyond doubt (at least eight evaluations after the best one,
slope p < 0.001), so a slow, steady onset is still caught early.

A run with only a handful of evaluations cannot pass a slope test. It is
still reported when its last evaluation is at least `error_rise` above the
best, or `far_rise` (default 25%) above with the evaluation before it also
at least `warn_rise` above: 21 public runs ended a quarter above their best
evaluation loss with too few evaluations for the trend test to say so. On public `Trainer` logs with an evaluation loss
(`studies/hub-census/`), about one run in five ends 5% or more above its
best and one in twenty ends 50% above; a 1% rise, the earlier bar for a
warning, is within what a small evaluation set does by chance.

When the training loss kept falling over the same range the finding is
titled "Overfitting onset"; when it did not, or the log has no training
loss, it is "Evaluation loss rising" and makes no claim about the cause.

**False-positive modes.**
- A noisy eval set (small validation split) can show a spurious upward
  eval trend after the true best point by chance; see the `eval_noise`
  detector, which flags exactly this risk and should be read alongside
  this one.
- A change in eval *data* mid-run (curriculum, harder eval subset added
  later) looks identical to genuine overfitting.
- LR schedule changes near `k*` (e.g. decay finishing) can cause both
  metrics to move in ways that coincidentally resemble this pattern.
- Divergence trivially satisfies this test too: an exploding eval loss is
  technically "getting worse after the best point." `detectors.run_all`
  downgrades a finding here to `info` when its step range overlaps a
  `divergence` finding, or an error-severity `spikes` finding, rather than
  reporting the same explosion twice under two different names -- calling
  this detector directly (not through `run_all`) skips that attribution.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, Severity, format_p, ols_slope, series_arrays


def _rise(fraction: float) -> str:
    """5.6% for an ordinary rise; a multiple once the percentage stops being readable."""
    return f"{fraction * 100:.1f}%" if fraction < 10 else f"{fraction + 1:.0f}x its minimum"


class OverfittingDetector(Detector):
    name = "overfitting"

    def __init__(
        self,
        train_metric: str = "train/loss",
        eval_metric: str = "eval/loss",
        alpha: float = 0.1,
        min_points_after: int = 3,
        min_rise: float = 0.03,
        warn_rise: float = 0.10,
        clear_rise: float = 0.05,
        far_rise: float = 0.25,
        error_rise: float = 0.50,
    ) -> None:
        self.train_metric = train_metric
        self.eval_metric = eval_metric
        self.alpha = alpha
        self.min_points_after = min_points_after
        self.min_rise = min_rise
        self.warn_rise = warn_rise
        self.clear_rise = clear_rise
        self.far_rise = far_rise
        self.error_rise = error_rise

    def _run(self, run: Run) -> list[Finding]:
        eval_series = run.get(self.eval_metric)
        if eval_series is None or len(eval_series) < 3:
            return []
        eval_steps, eval_values = series_arrays(eval_series)
        best_i = int(np.argmin(eval_values))
        after = eval_values[best_i:]
        after_steps = eval_steps[best_i:]
        tail = after[1:]
        if len(tail) == 0:
            return []

        best_value = float(eval_values[best_i])
        current_value = float(eval_values[-1])
        rise = (current_value - best_value) / (abs(best_value) + 1e-12)
        slope, p_value = 0.0, 1.0
        if len(tail) >= self.min_points_after:
            slope, _se, p_value = ols_slope(after_steps, after)
        trend = slope > 0 and p_value < self.alpha and rise >= self.min_rise
        # With three or four evaluations in the whole run no slope can be significant.
        # A last evaluation far above the best one is still worth saying: half again
        # above it, or a quarter above with the one before it also clearly higher.
        far_above = rise >= self.error_rise or (
            rise >= self.far_rise
            and len(tail) >= 2
            and float(tail[-2]) >= best_value * (1 + self.warn_rise)
        )
        if not (trend or far_above):
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

        # A small rise is still a warning when the trend is beyond doubt: catching the
        # onset early is the point, and eight evaluations rising at p < 0.001 is not noise.
        clear_trend = rise >= self.clear_rise and p_value < 0.001 and len(after) - 1 >= 8
        evidence = (
            f"slope {format_p(p_value)}"
            if trend
            else f"over {len(tail)} evaluation{'s' if len(tail) != 1 else ''}, too few for a trend test"
        )
        severity: Severity = (
            "error"
            if rise >= self.error_rise
            else "warning"
            if rise >= self.warn_rise or clear_trend
            else "info"
        )
        return [
            Finding(
                detector=self.name,
                # Without a falling training loss beside it, a rising evaluation loss is
                # only that: there may be no training loss in the log at all.
                title="Overfitting onset" if train_note else "Evaluation loss rising",
                severity=severity,
                metric=self.eval_metric,
                step_start=int(eval_steps[best_i]),
                step_end=int(eval_steps[-1]),
                message=(
                    f"Best {self.eval_metric} was {best_value:.4g} at step "
                    f"{int(eval_steps[best_i])}; it's since risen to {current_value:.4g} "
                    f"({_rise(rise)}, {evidence})."
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
