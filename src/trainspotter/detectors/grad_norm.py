"""Gradient-norm detector: explosions and clipping saturation.

Only runs when a `grad_norm` metric is logged (most training loops that
clip gradients log the norm *before* clipping; trainspotter doesn't know
which convention was used and says so in the finding).

**Explosion algorithm.** A reading is an explosion when it is both unusual
(the spike detector's robust z-score, `window` default 21, above
`threshold`, default 10) and large (at least `min_ratio`, default 10, times
the rolling median around it). All of a run's explosions are reported as
one `info` finding with their count and the largest. Gradient norms are
heavy-tailed: on 3,145 public `Trainer` logs (`studies/hub-census/`) the
z-score alone fired on four runs in ten, 24 times each on average, and
seven in ten of those runs ended at their best loss. Whether an explosion
mattered is a question for the loss, so this finding says so and leaves the
severity to the spike, setback and divergence detectors. A NaN or infinite
norm is reported separately, also as `info`: 46 public runs that ended at
their best loss logged one.

**Clipping-saturation algorithm.** Over a trailing window of `sat_window`
(default 50) points, compute how many are within `sat_tol` (default 0.5%)
of the window's own maximum value. If that fraction exceeds
`sat_fraction` (default 0.8), the norm is spending most of its time pinned
at (or just under) one ceiling value -- almost always the clip threshold,
meaning clipping is doing most of the work of controlling gradient scale
rather than occasional correction.

**False-positive modes.**
- If `grad_norm` is logged *after* clipping, it's naturally capped at the
  clip value and saturation there is by design, not a problem -- treat
  this finding as informational unless you know norms are logged
  pre-clip.
- A model/task with a genuinely constant gradient scale (rare, but
  possible near convergence) can trip saturation without any clipping
  involved.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import (
    Detector,
    Finding,
    FloatArray,
    group_consecutive,
    robust_zscore,
    rolling_median_mad,
    series_arrays,
)

DEFAULT_WINDOW = 21
DEFAULT_THRESHOLD = 10.0


class GradNormDetector(Detector):
    name = "grad_norm"

    def __init__(
        self,
        metric: str = "grad_norm",
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
        min_ratio: float = 10.0,
        sat_window: int = 50,
        sat_tol: float = 0.005,
        sat_fraction: float = 0.8,
    ) -> None:
        self.metric = metric
        self.window = window
        self.threshold = threshold
        self.min_ratio = min_ratio
        self.sat_window = sat_window
        self.sat_tol = sat_tol
        self.sat_fraction = sat_fraction

    def _run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < 5:
            return []
        steps, values = series_arrays(series)
        findings = self._explosions(steps, values)
        findings.extend(self._saturation(steps, values))
        return findings

    def _explosions(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        findings: list[Finding] = []
        finite = np.isfinite(values)
        if not finite.all():
            bad = np.flatnonzero(~finite)
            findings.append(
                Finding(
                    detector=self.name,
                    title="Non-finite gradient norm",
                    # Mixed-precision training logs these when the scaler skips a step. If
                    # the loss went non-finite too, the divergence detector says so.
                    severity="info",
                    metric=self.metric,
                    step_start=int(steps[bad[0]]),
                    step_end=int(steps[bad[-1]]),
                    message=(
                        f"grad_norm was NaN or infinite at {len(bad)} logged step"
                        f"{'s' if len(bad) != 1 else ''}, first at step {int(steps[bad[0]])}."
                    ),
                    evidence={"first_bad_step": int(steps[bad[0]]), "count": int(len(bad))},
                    fixes=[
                        "If the loss stayed finite, this is usually mixed-precision overflow in the norm itself: try bf16, or check the gradient scaler.",
                        "If the loss went non-finite too, see the divergence finding.",
                    ],
                )
            )
        steps, values = steps[finite], values[finite]
        if len(values) < 5:
            return findings
        z = robust_zscore(values, self.window)
        median, _mad = rolling_median_mad(values, self.window)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(median > 0, values / median, 0.0)
        flagged = [i for i in range(len(values)) if z[i] > self.threshold and ratio[i] >= self.min_ratio]
        groups = group_consecutive(flagged, gap=2)
        if not groups:
            return findings
        peak_i = max(flagged, key=lambda i: ratio[i])
        findings.append(
            Finding(
                detector=self.name,
                title="Gradient-norm explosion",
                severity="info",
                metric=self.metric,
                step_start=int(steps[groups[0][0]]),
                step_end=int(steps[groups[-1][-1]]),
                message=(
                    f"grad_norm exceeded {self.min_ratio:g}x its rolling median "
                    f"{len(groups)} time{'s' if len(groups) != 1 else ''}; the largest was "
                    f"{values[peak_i]:.4g} at step {int(steps[peak_i])}, {ratio[peak_i]:.0f}x the "
                    "median around it. Read it with the loss: an explosion the loss did not "
                    "react to was absorbed (usually by clipping)."
                ),
                evidence={
                    "count": len(groups),
                    "peak_step": int(steps[peak_i]),
                    "peak_value": float(values[peak_i]),
                    "ratio_to_median": float(ratio[peak_i]),
                },
                fixes=[
                    "Enable or lower gradient clipping (e.g. max-norm 1.0).",
                    "Lower the learning rate.",
                    "Check the batch at this step for an outlier example (e.g. a corrupted/extreme-length sample).",
                ],
            )
        )
        return findings

    def _saturation(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        n = len(values)
        if n < self.sat_window:
            return []
        window = values[-self.sat_window :]
        ceiling = float(np.max(window))
        if ceiling <= 0:
            return []
        near_ceiling = np.abs(window - ceiling) <= self.sat_tol * ceiling
        fraction = float(np.mean(near_ceiling))
        if fraction < self.sat_fraction:
            return []
        return [
            Finding(
                detector=self.name,
                title="Gradient clipping likely saturated",
                severity="info",
                metric=self.metric,
                step_start=int(steps[-self.sat_window]),
                step_end=int(steps[-1]),
                message=(
                    f"{fraction * 100:.0f}% of the last {self.sat_window} grad_norm readings "
                    f"sit within {self.sat_tol * 100:.1f}% of {ceiling:.4g} -- consistent with "
                    "clipping pinning the norm at a ceiling most steps."
                ),
                evidence={"ceiling": ceiling, "fraction_at_ceiling": fraction, "window": self.sat_window},
                fixes=[
                    "If grad_norm is logged pre-clip: raise the clip threshold, or lower the LR so gradients naturally stay under it.",
                    "If grad_norm is logged post-clip, this is expected by construction and not necessarily a problem.",
                ],
            )
        ]
