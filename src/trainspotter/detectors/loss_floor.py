"""Loss-floor / suspected-leakage heuristic.

**This is explicitly a heuristic, not a statistical test.** A curve alone
cannot establish that a loss is *too* low without knowing the task, the loss function's theoretical
minimum, and the label distribution. A fast drop to near-zero loss is a
reason to inspect those conditions, not evidence of leakage by itself.

**Algorithm.** Within the first `early_frac` (default 10%) of the interval
between the first and last logged training steps, check whether the metric
has already dropped below `abs_floor`
(default 0.05, tuned for a typical cross-entropy loss on a non-trivial
task -- override for MSE, contrastive, or other losses with a different
natural scale) AND has fallen to at most `rel_floor` (default 5%) of its
first logged value. Both conditions together avoid flagging a task whose
loss is simply small in absolute terms from the start (e.g. a well-scaled
regression target).

The window uses training coordinates, not the number of logged observations:
changing logging cadence must not turn a late observation into an early one.
Row indices and W&B logging counters cannot establish training progress, so
this heuristic requires a training-step axis. The first value must be finite
and positive, and the early window must contain only finite, nonnegative
readings. A missing/nonfinite baseline cannot establish a relative drop.
The observed interval may start at a resumed checkpoint or end before the
run is complete; this is never evidence about an unobserved full run.

**False-positive modes (real, not edge cases -- read this list before
trusting a finding here).**
- A genuinely easy task (small vocabulary, near-deterministic mapping,
  tiny/simple dataset the model can fit almost exactly) legitimately
  reaches very low loss fast -- this is not leakage.
- A loss with a naturally small scale (normalized MSE, cosine-similarity
  losses near their optimum) trips the absolute floor without any
  leakage.
- Resuming from a checkpoint that was already well-trained produces a low
  loss from step 0 of *this* log by construction.

Treat a finding here as "worth a five-minute look at the data pipeline
for train/eval overlap or label leakage," not as a verdict.
"""

from __future__ import annotations

import math

from trainspotter.model import Run

from .base import Detector, Finding

DEFAULT_EARLY_FRAC = 0.10
DEFAULT_ABS_FLOOR = 0.05
DEFAULT_REL_FLOOR = 0.05


class LossFloorDetector(Detector):
    name = "loss_floor"

    def __init__(
        self,
        metric: str = "train/loss",
        early_frac: float = DEFAULT_EARLY_FRAC,
        abs_floor: float = DEFAULT_ABS_FLOOR,
        rel_floor: float = DEFAULT_REL_FLOOR,
        min_points: int = 5,
    ) -> None:
        self.metric = metric
        self.early_frac = early_frac
        self.abs_floor = abs_floor
        self.rel_floor = rel_floor
        self.min_points = min_points

    def _run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < self.min_points or run.axis_kind() != "step":
            return []
        first, last = series.points[0], series.points[-1]
        span = last.step - first.step
        if span <= 0 or not math.isfinite(first.value) or first.value <= 0:
            return []
        early = [p for p in series.points if p.step - first.step <= span * self.early_frac]
        # Validate the whole window before claiming an early drop: no nonfinite
        # gaps, or a negative objective whose floor is not zero.
        if any(not math.isfinite(p.value) or p.value < 0 for p in early):
            return []
        for point in early:
            v = point.value
            if v <= self.abs_floor and v <= first.value * self.rel_floor:
                position = (point.step - first.step) / span
                return [
                    Finding(
                        detector=self.name,
                        title="Suspiciously low loss early (heuristic)",
                        severity="info",
                        metric=self.metric,
                        step_start=first.step,
                        step_end=point.step,
                        message=(
                            f"{self.metric} dropped to {v:.4g} by step {point.step} -- "
                            f"only {v / first.value * 100:.1f}% of its first logged value "
                            f"({first.value:.4g}), at {position:.1%} of the logged step span "
                            f"({first.step}-{last.step}). Possibly just an "
                            "easy task; possibly train/eval overlap or label leakage. "
                            "This heuristic does not establish leakage or describe "
                            "training outside the logged interval."
                        ),
                        evidence={
                            "step": point.step,
                            "value": v,
                            "first_value": first.value,
                            "fraction_of_start": v / first.value,
                            "first_logged_step": first.step,
                            "last_logged_step": last.step,
                            "early_span_fraction": self.early_frac,
                            "fraction_of_logged_span": position,
                        },
                        fixes=[
                            "Check for train/eval overlap (duplicate or near-duplicate examples).",
                            "Check the label isn't leaking into the input features (e.g. a column that encodes the target).",
                            "If the task is genuinely easy (small/simple dataset), this may be entirely normal.",
                        ],
                    )
                ]
        return []
