"""Loss-floor / suspected-leakage heuristic.

**This is explicitly a heuristic, not a statistical test** -- unlike the
other detectors, there's no distribution-free way to say "this loss is
*too* low" without knowing the task, the loss function's theoretical
minimum, and the label distribution. It exists because an implausibly
fast drop to near-zero loss is one of the most common early symptoms an
experienced engineer would eyeball and immediately distrust.

**Algorithm.** Within the first `early_frac` (default 10%) of logged
steps, check whether the metric has already dropped below `abs_floor`
(default 0.05, tuned for a typical cross-entropy loss on a non-trivial
task -- override for MSE, contrastive, or other losses with a different
natural scale) AND has fallen to at most `rel_floor` (default 5%) of its
first logged value. Both conditions together avoid flagging a task whose
loss is simply small in absolute terms from the start (e.g. a well-scaled
regression target).

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

from trainspotter.model import Run

from .base import Detector, Finding, series_arrays

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

    def run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < self.min_points:
            return []
        steps, values = series_arrays(series)
        n = len(values)
        early_n = max(1, int(n * self.early_frac))
        first_value = float(values[0])
        if first_value <= 0:
            return []
        for i in range(early_n):
            v = float(values[i])
            if v <= self.abs_floor and v <= first_value * self.rel_floor:
                return [
                    Finding(
                        detector=self.name,
                        title="Suspiciously low loss early (heuristic)",
                        severity="info",
                        metric=self.metric,
                        step_start=int(steps[0]),
                        step_end=int(steps[i]),
                        message=(
                            f"{self.metric} dropped to {v:.4g} by step {int(steps[i])} -- "
                            f"only {v / first_value * 100:.1f}% of its starting value "
                            f"({first_value:.4g}), this early in the run. Possibly just an "
                            "easy task; possibly train/eval overlap or label leakage. "
                            "Heuristic, not a statistical finding -- see docstring for "
                            "false-positive modes."
                        ),
                        evidence={
                            "step": int(steps[i]),
                            "value": v,
                            "first_value": first_value,
                            "fraction_of_start": v / first_value,
                        },
                        fixes=[
                            "Check for train/eval overlap (duplicate or near-duplicate examples).",
                            "Check the label isn't leaking into the input features (e.g. a column that encodes the target).",
                            "If the task is genuinely easy (small/simple dataset), this may be entirely normal.",
                        ],
                    )
                ]
        return []
