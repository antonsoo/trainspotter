"""Gradient-norm detector: explosions and clipping saturation.

Only runs when a `grad_norm` metric is logged (most training loops that
clip gradients log the norm *before* clipping; trainspotter doesn't know
which convention was used and says so in the finding).

**Explosion algorithm.** Same robust z-score as the spike detector (see
`spikes.py`), applied to `grad_norm` with `window` (default 21) and
`threshold` (default 6.0).

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

import math

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, group_consecutive, robust_zscore, series_arrays

DEFAULT_WINDOW = 21
DEFAULT_THRESHOLD = 6.0


class GradNormDetector(Detector):
    name = "grad_norm"

    def __init__(
        self,
        metric: str = "grad_norm",
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
        sat_window: int = 50,
        sat_tol: float = 0.005,
        sat_fraction: float = 0.8,
    ) -> None:
        self.metric = metric
        self.window = window
        self.threshold = threshold
        self.sat_window = sat_window
        self.sat_tol = sat_tol
        self.sat_fraction = sat_fraction

    def run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < 5:
            return []
        steps, values = series_arrays(series)
        findings = self._explosions(steps, values)
        findings.extend(self._saturation(steps, values))
        return findings

    def _explosions(self, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
        z = robust_zscore(values, self.window)
        flagged = [i for i in range(len(values)) if z[i] > self.threshold]  # one-sided: only spikes up
        findings = []
        for group in group_consecutive(flagged, gap=2):
            peak_i = max(group, key=lambda i: z[i])
            peak_z = z[peak_i]
            scale_note = (
                "far beyond (non-finite) the local MAD scale above its rolling median"
                if not math.isfinite(peak_z)
                else f"{peak_z:.1f}x the local MAD scale above its rolling median"
            )
            findings.append(
                Finding(
                    detector=self.name,
                    title="Gradient-norm explosion",
                    severity="warning",
                    metric=self.metric,
                    step_start=int(steps[group[0]]),
                    step_end=int(steps[group[-1]]),
                    message=(
                        f"grad_norm hit {values[peak_i]:.4g} at step {int(steps[peak_i])}, {scale_note}."
                    ),
                    evidence={"peak_step": int(steps[peak_i]), "peak_value": float(values[peak_i]), "robust_z": float(peak_z)},
                    fixes=[
                        "Enable or lower gradient clipping (e.g. max-norm 1.0).",
                        "Lower the learning rate.",
                        "Check the batch at this step for an outlier example (e.g. a corrupted/extreme-length sample).",
                    ],
                )
            )
        return findings

    def _saturation(self, steps: np.ndarray, values: np.ndarray) -> list[Finding]:
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
