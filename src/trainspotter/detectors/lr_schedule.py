"""Learning-rate schedule anomaly detector.

Looks at the `lr` metric only (no assumptions about what schedule was
*intended* -- trainspotter never sees the config, only the logged values)
and runs three independent checks:

1. **Missing warmup.** If the LR's first logged value is already within
   `warmup_frac` (default 90%) of the run's peak LR, and the run has more
   than `min_steps_for_warmup_check` (default 20) points, there was
   effectively no ramp-up. Reported as `info`: it is a configuration
   choice (and `transformers.Trainer`'s default), not a fault.
2. **Rise after decay.** Walk the observations in time order. A decline
   exceeding `bump_tol` (default 1% of its preceding peak) establishes a
   falling phase. A subsequent rise exceeding the same fraction of the
   larger of the preceding peak and the new value establishes a rebound.
   Report its preceding peak, low point, first confirmation and high point,
   even if it never reaches the preceding peak. Small wiggles and plateaus
   stay in the same phase; another material decline starts a new phase.
   Optional `peak_grace_frac` ignores lows inside an initial fraction of
   logged points. It now defaults to zero: the decline check separates a
   rebound from warmup, and appending later points must not erase an early
   rebound. An explicit grace uses point count, not elapsed steps.
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
   instead sidesteps that failure mode entirely. An upward discontinuity
   already covered by a rebound is reported only as that rebound.

**False-positive modes.**
- Cyclic schedules (cosine with restarts, one-cycle repeated, cyclical
  LR) are *designed* to be non-monotonic; checks 2 and 3 will fire
  repeatedly. All findings here are informational: the log cannot tell
  whether a rise was intended, a resume mistake or an overrun of a schedule.
- A schedule with genuinely few logged points (coarse logging interval)
  can have single steps that cover a large fraction of the LR range by
  construction, not because anything broke. Check 3 only reports a jump
  that stands out from its neighbors: if an adjacent logged interval moves
  the same way at no less than half the per-step rate, the jump is one leg
  of a steady ramp and isn't reported. A rise before the LR has ever
  fallen is never reported: that is warmup, and a short one logged every
  few steps covers the whole range in two or three uneven intervals.
- Step-decay schedules (``MultiStepLR``-style drops) change the LR in
  deliberate single jumps; check 3 reports each drop, as ``info``.
"""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, FloatArray, group_consecutive, series_arrays


def _between(step_gap: int) -> str:
    return "in a single step" if step_gap <= 1 else f"across one logged interval ({step_gap} steps)"


class LRScheduleDetector(Detector):
    name = "lr_schedule"

    def __init__(
        self,
        metric: str = "lr",
        warmup_frac: float = 0.9,
        min_steps_for_warmup_check: int = 20,
        bump_tol: float = 0.01,
        peak_grace_frac: float = 0.0,
        jump_frac: float = 0.2,
    ) -> None:
        self.metric = metric
        self.warmup_frac = warmup_frac
        self.min_steps_for_warmup_check = min_steps_for_warmup_check
        self.bump_tol = bump_tol
        self.peak_grace_frac = peak_grace_frac
        self.jump_frac = jump_frac

    def _run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < 3:
            return []
        steps, values = series_arrays(series)
        findings: list[Finding] = []
        findings.extend(self._missing_warmup(steps, values))
        rebounds = self._rises_after_decay(steps, values)
        findings.extend(rebounds)
        starts = [f.step_start for f in rebounds]
        for jump in self._discontinuity(steps, values):
            prior = bisect_right(starts, jump.step_start) - 1
            rising = float(jump.evidence["after"]) > float(jump.evidence["before"])
            if rising and prior >= 0 and jump.step_end <= rebounds[prior].step_end:
                continue
            findings.append(jump)
        return findings

    def _missing_warmup(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
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
                    # A configuration choice, and `transformers.Trainer`'s default: one in six
                    # public runs that log a learning rate show no warmup (studies/hub-census).
                    severity="info",
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
                        "If the first steps are unstable (see any spike or setback findings near the start), add LR warmup over the first 1-10% of steps.",
                        "If warmup is configured but not logged until after it finishes, this may be a logging artifact -- check the logging interval.",
                    ],
                )
            ]
        return []

    def _rises_after_decay(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        grace = int(len(values) * self.peak_grace_frac)
        findings: list[Finding] = []
        peak: int | None = None
        low: int | None = None
        # Indices of the preceding peak, the low, and the first confirmed rise.
        # Once confirmed, `peak` follows this episode's high until a material fall.
        episode: tuple[int, int, int] | None = None

        def finish() -> None:
            if episode is None or peak is None:
                return
            previous, bottom, confirmed = episode
            if bottom < grace:
                return
            earlier = float(values[previous])
            minimum = float(values[bottom])
            maximum = float(values[peak])
            findings.append(
                Finding(
                    detector=self.name,
                    title="LR rose after decay",
                    severity="info",
                    metric=self.metric,
                    step_start=int(steps[bottom]),
                    step_end=int(steps[peak]),
                    message=(
                        f"{self.metric} fell from {earlier:.4g} at step {int(steps[previous])} "
                        f"to {minimum:.4g} at step {int(steps[bottom])}, then rose to "
                        f"{maximum:.4g} at step {int(steps[peak])}. "
                        "This can be an intended cycle or restart; the log alone does not "
                        "establish a scheduler fault."
                    ),
                    evidence={
                        "earlier_peak": earlier,
                        "peak_step": int(steps[previous]),
                        "low_value": minimum,
                        "low_step": int(steps[bottom]),
                        "high_value": maximum,
                        "high_step": int(steps[peak]),
                        "confirmed_step": int(steps[confirmed]),
                        "rise_fraction_of_peak": (maximum - minimum) / max(earlier, maximum),
                        "bump_tolerance": self.bump_tol,
                    },
                    fixes=[
                        "If this matches an intended cycle or warm restart, no correction is needed.",
                        "Otherwise, check that scheduler state was restored with the optimizer after a checkpoint resume.",
                        "Check the scheduler's configured horizon and how often scheduler.step() is called.",
                    ],
                )
            )

        for i, raw_value in enumerate(values):
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                # Do not join a decline and a rise across an unusable observation.
                # Already observed episodes remain reportable before the gap.
                finish()
                peak, low, episode = None, None, None
            elif peak is None:
                peak = i
            elif low is None:
                peak_value = float(values[peak])
                if value >= peak_value:
                    peak = i
                elif peak_value - value > self.bump_tol * peak_value:
                    finish()
                    episode = None
                    low = i
            elif value <= values[low]:
                low = i
            elif value - float(values[low]) > self.bump_tol * max(float(values[peak]), value):
                episode = peak, low, i
                peak, low = i, None
        finish()
        return findings

    def _discontinuity(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        if len(values) < 6:
            return []
        lr_range = float(np.max(values) - np.min(values))
        if lr_range <= 0:
            return []
        deltas = np.diff(values)
        # Per-step rate, so a neighbor logged over a different interval compares fairly.
        rates = deltas / np.maximum(np.diff(steps), 1)
        threshold = self.jump_frac * lr_range

        def part_of_ramp(i: int) -> bool:
            for j in (i - 1, i + 1):
                same_way = 0 <= j < len(rates) and np.sign(rates[j]) == np.sign(rates[i])
                if same_way and abs(rates[j]) >= 0.5 * abs(rates[i]):
                    return True
            return False

        # A rise before the LR has ever fallen is warmup, however coarsely it was logged: a
        # ten-step warmup logged every five steps is two jumps of half the range each.
        falls = np.flatnonzero(deltas < 0)
        first_fall = int(falls[0]) if len(falls) else len(deltas)

        def is_warmup(i: int) -> bool:
            return bool(deltas[i] > 0 and i < first_fall)

        flagged = [
            i
            for i in range(len(deltas))
            if abs(deltas[i]) > threshold and not part_of_ramp(i) and not is_warmup(i)
        ]
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
                        f"({lr_range:.4g}) {_between(int(steps[i + 1] - steps[i]))}, out of line "
                        "with the logged intervals around it."
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
