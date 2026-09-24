"""Divergence detector.

**Algorithm.** Two independent triggers, either sufficient:

1. *Non-finite*: any logged value is NaN or +/-Inf. Always `error`
   severity -- there is no ambiguity here, training has broken.
2. *Sustained growth*: fit an OLS line to the last `tail_frac` (default
   25%) of the run (minimum `min_window` points, default 10) and flag if
   the slope is positive with p < `alpha` (default 0.05, via
   `detectors.base.ols_slope`'s normal-approximation t-test) AND the last
   value is at least `growth_ratio` (default 1.5x) above the run's running
   minimum. Both conditions matter: a positive-but-tiny slope on a metric
   that's still near its best value is noise, not divergence.

**False-positive modes.**
- A metric that's supposed to increase (accuracy, reward, a "goodness"
  eval score) will trip the sustained-growth trigger for the *right*
  reason -- point this detector only at metrics where lower is better
  (its default, `train/loss` and `eval/loss`).
- A cyclic LR schedule or a curriculum that temporarily increases task
  difficulty can produce a real, temporary upward trend that recovers
  later; this detector only looks at the tail window, so it can't
  distinguish "still recovering" from "diverged for good" without more
  context.
"""

from __future__ import annotations

import math

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, ols_slope, series_arrays

DEFAULT_METRICS = ("train/loss", "eval/loss")


class DivergenceDetector(Detector):
    name = "divergence"

    def __init__(
        self,
        metrics: tuple[str, ...] = DEFAULT_METRICS,
        tail_frac: float = 0.25,
        min_window: int = 10,
        alpha: float = 0.05,
        growth_ratio: float = 1.5,
    ) -> None:
        self.metrics = metrics
        self.tail_frac = tail_frac
        self.min_window = min_window
        self.alpha = alpha
        self.growth_ratio = growth_ratio

    def run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None or len(series) < 3:
                continue
            steps, values = series_arrays(series)
            findings.extend(self._non_finite(metric_name, steps, values))
            findings.extend(self._sustained_growth(metric_name, steps, values))
        return findings

    def _non_finite(self, metric_name: str, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
        bad = [i for i in range(len(values)) if not math.isfinite(values[i])]
        if not bad:
            return []
        first, last = bad[0], bad[-1]
        kinds = sorted({"NaN" if math.isnan(values[i]) else "Inf" for i in bad})
        return [
            Finding(
                detector=self.name,
                title="Diverged (non-finite values)",
                severity="error",
                metric=metric_name,
                step_start=int(steps[first]),
                step_end=int(steps[last]),
                message=(
                    f"{metric_name} became {'/'.join(kinds)} at step {int(steps[first])} "
                    f"({len(bad)} non-finite point(s) in this run)."
                ),
                evidence={"first_bad_step": int(steps[first]), "count": len(bad)},
                fixes=[
                    "Lower the learning rate; this is the single most common cause.",
                    "Enable gradient clipping if it isn't already on.",
                    "Check for a divide-by-zero or log(0) in the loss (e.g. an unstable custom loss term, an empty batch after filtering).",
                    "If using mixed precision, check the loss-scaler didn't overflow, or try bf16 instead of fp16.",
                ],
            )
        ]

    def _sustained_growth(
        self, metric_name: str, steps: np.ndarray, values: np.ndarray
    ) -> list[Finding]:
        finite = np.isfinite(values)
        if finite.sum() < self.min_window:
            return []
        steps_f, values_f = steps[finite], values[finite]
        n = len(values_f)
        window = max(self.min_window, int(n * self.tail_frac))
        if window >= n:
            return []
        tail_steps, tail_values = steps_f[-window:], values_f[-window:]
        slope, _se, p_value = ols_slope(tail_steps, tail_values)
        running_min = float(np.min(values_f))
        last_value = float(tail_values[-1])
        if running_min <= 0:  # noqa: SIM108 (nested ternary here is less readable)
            ratio = float("inf") if last_value > 0 else 1.0
        else:
            ratio = last_value / running_min
        if slope > 0 and p_value < self.alpha and ratio >= self.growth_ratio:
            return [
                Finding(
                    detector=self.name,
                    title="Diverged (sustained growth)",
                    severity="error",
                    metric=metric_name,
                    step_start=int(tail_steps[0]),
                    step_end=int(tail_steps[-1]),
                    message=(
                        f"{metric_name} has been climbing for the last {window} logged points "
                        f"(slope p={p_value:.3g}) and is now {ratio:.2f}x its running minimum "
                        f"({running_min:.4g})."
                    ),
                    evidence={
                        "slope": slope,
                        "p_value": p_value,
                        "running_min": running_min,
                        "last_value": last_value,
                        "ratio_to_min": ratio,
                        "window": window,
                    },
                    fixes=[
                        "Lower the learning rate or restart from the last good checkpoint before this window.",
                        "Check the LR schedule isn't re-warming up unintentionally (see lr_schedule detector).",
                        "Look for a data distribution shift at this step range (new shard, curriculum stage change).",
                    ],
                )
            ]
        return []
