"""Learning-rate schedule anomaly detector.

Looks at the `lr` metric only (no assumptions about what schedule was
*intended* -- trainspotter never sees the config, only the logged values)
and runs three independent checks:

1. **Missing warmup.** If the LR's first logged value is already within
   `warmup_frac` (default 90%) of the run's peak LR, and the run has more
   than `min_steps_for_warmup_check` (default 20) points, there was
   effectively no ramp-up.
2. **Non-monotonic after peak.** Once LR has reached its running maximum,
   almost every schedule (step, cosine, linear, polynomial decay) is
   non-increasing from then on. A later value that exceeds the running
   maximum by more than `bump_tol` (default 1%) -- outside the first
   `peak_grace_frac` (default 5%) of steps, to allow a slightly noisy
   peak -- is flagged as unexpected.
3. **Discontinuity.** A step-to-step LR change whose absolute size exceeds
   `jump_frac` (default 20%) of the LR's whole observed range
   (`max(lr) - min(lr)` over the run) -- a jump big enough, relative to
   the schedule's entire span, that it can't be one ordinary step of a
   smooth ramp or decay. This deliberately avoids a rolling-MAD/z-score
   test: most schedules spend long stretches flat (full warmup, or a
   constant LR phase), which makes a *local* variability estimate
   collapse to ~0 and then call the very next real step of the ramp an
   "outlier" -- a false positive trainspotter hit on its own warmup demo
   data during development. Measuring jumps against the run's total range
   instead sidesteps that failure mode entirely.

**False-positive modes.**
- Cyclic schedules (cosine with restarts, one-cycle repeated, cyclical
  LR) are *designed* to be non-monotonic; checks 2 and 3 will fire
  repeatedly and should be ignored for those schedules.
- A schedule with genuinely few logged points (coarse logging interval)
  can have single steps that cover a large fraction of the LR range by
  construction, not because anything broke.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, group_consecutive, series_arrays


class LRScheduleDetector(Detector):
    name = "lr_schedule"

    def __init__(
        self,
        metric: str = "lr",
        warmup_frac: float = 0.9,
        min_steps_for_warmup_check: int = 20,
        bump_tol: float = 0.01,
        peak_grace_frac: float = 0.05,
        jump_frac: float = 0.2,
    ) -> None:
        self.metric = metric
        self.warmup_frac = warmup_frac
        self.min_steps_for_warmup_check = min_steps_for_warmup_check
        self.bump_tol = bump_tol
        self.peak_grace_frac = peak_grace_frac
        self.jump_frac = jump_frac

    def run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < 5:
            return []
        steps, values = series_arrays(series)
        findings: list[Finding] = []
        findings.extend(self._missing_warmup(steps, values))
        findings.extend(self._non_monotonic_after_peak(steps, values))
        findings.extend(self._discontinuity(steps, values))
        return findings

    def _missing_warmup(self, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
        if len(values) < self.min_steps_for_warmup_check:
            return []
        peak = float(np.max(values))
        if peak <= 0:
            return []
        if values[0] / peak >= self.warmup_frac:
            return [
                Finding(
                    detector=self.name,
                    title="No LR warmup detected",
                    severity="warning",
                    metric=self.metric,
                    step_start=int(steps[0]),
                    step_end=int(steps[0]),
                    message=(
                        f"lr starts at {values[0]:.4g}, already "
                        f"{values[0] / peak * 100:.0f}% of the run's peak "
                        f"({peak:.4g}) -- no ramp-up phase visible in the log."
                    ),
                    evidence={"initial_lr": float(values[0]), "peak_lr": peak},
                    fixes=[
                        "Add LR warmup (e.g. linear warmup over the first 1-10% of steps) -- it's a common fix for early spikes/instability.",
                        "If warmup is configured but not logged until after it finishes, this may be a logging artifact -- check the logging interval.",
                    ],
                )
            ]
        return []

    def _non_monotonic_after_peak(self, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
        n = len(values)
        peak_i = int(np.argmax(values))
        grace = int(n * self.peak_grace_frac)
        start = max(peak_i + 1, grace)
        if start >= n:
            return []
        running_max = values[peak_i]
        peak_val = float(values[peak_i])
        bumps = []
        for i in range(start, n):
            running_max = max(running_max, values[i])
            if values[i] > peak_val * (1 + self.bump_tol) and i > peak_i:
                bumps.append(i)
        if not bumps:
            return []
        i = bumps[0]
        return [
            Finding(
                detector=self.name,
                title="LR increased after its peak",
                severity="info",
                metric=self.metric,
                step_start=int(steps[i]),
                step_end=int(steps[bumps[-1]]),
                message=(
                    f"lr rose to {values[i]:.4g} at step {int(steps[i])}, above its "
                    f"earlier peak of {peak_val:.4g} at step {int(steps[peak_i])}. "
                    "Expected for cyclic/restart schedules; unexpected otherwise."
                ),
                evidence={"earlier_peak": peak_val, "peak_step": int(steps[peak_i])},
                fixes=[
                    "If you're using a cyclic schedule or warm restarts, this is expected -- ignore.",
                    "Otherwise, check the scheduler's state was restored correctly after a checkpoint resume.",
                ],
            )
        ]

    def _discontinuity(self, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
        if len(values) < 6:
            return []
        lr_range = float(np.max(values) - np.min(values))
        if lr_range <= 0:
            return []
        deltas = np.diff(values)
        threshold = self.jump_frac * lr_range
        flagged = [i for i in range(len(deltas)) if abs(deltas[i]) > threshold]
        findings = []
        for group in group_consecutive(flagged, gap=1):
            i = max(group, key=lambda i: abs(deltas[i]))
            findings.append(
                Finding(
                    detector=self.name,
                    title="LR discontinuity",
                    severity="info",
                    metric=self.metric,
                    step_start=int(steps[i]),
                    step_end=int(steps[i + 1]),
                    message=(
                        f"lr jumped from {values[i]:.4g} to {values[i + 1]:.4g} between "
                        f"step {int(steps[i])} and {int(steps[i + 1])} -- "
                        f"{abs(deltas[i]) / lr_range * 100:.0f}% of the run's whole LR range "
                        f"({lr_range:.4g}) in a single step."
                    ),
                    evidence={
                        "before": float(values[i]),
                        "after": float(values[i + 1]),
                        "jump_fraction_of_range": abs(deltas[i]) / lr_range,
                    },
                    fixes=[
                        "Check for a checkpoint resume where the scheduler's step counter didn't match the optimizer's.",
                        "Check for a manual LR override mid-run.",
                    ],
                )
            )
        return findings
