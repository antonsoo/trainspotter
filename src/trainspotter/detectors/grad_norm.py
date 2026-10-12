"""Gradient-norm observations: nonfinite values, explosions, and concentration.

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

Pointwise nonfinite evidence needs only one reading and survives conflicting
repeated-step values. Explosions require at least five finite observations
in a contiguous segment; a NaN/Inf gap must not join unrelated windows.

**Concentration algorithm.** Over a trailing window of `sat_window`
(default 50) points, compute how many are within `sat_tol` (default 0.5%)
of the window's own maximum value. If that fraction exceeds
`sat_fraction` (default 0.8), report that observed concentration. The entire
window must be finite and nonnegative, with a positive maximum.

**Interpretation.** A constant gradient can occur without any clipping.
Post-clip norms can be capped by construction. Pre-clip norms do not measure
how often clipping changed an update. A norm history alone cannot identify
the logging convention, clipping threshold, or clipping frequency. This is
a descriptive note; measure before/after norms or a clipping indicator
before attributing the pattern to clipping or changing its threshold.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import (
    Detector,
    Finding,
    FloatArray,
    finite_spans,
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
        ambiguous = series is None and self.metric in run._excluded_metrics
        if ambiguous:
            series = run._excluded_metrics[self.metric]
        if series is None or series.is_empty():
            return []
        steps, values = series_arrays(series)
        observed = run.observed_series(self.metric)
        findings = []
        if observed is not None:
            observed_steps, observed_values = (
                (steps, values) if observed is series else series_arrays(observed)
            )
            findings = self._non_finite(observed_steps, observed_values, ambiguous=ambiguous)
        if ambiguous:
            return findings
        findings.extend(self._explosions(steps, values))
        findings.extend(self._saturation(steps, values))
        return findings

    def _non_finite(
        self, steps: FloatArray, values: FloatArray, *, ambiguous: bool,
    ) -> list[Finding]:
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
                        f"{self.metric} has {len(bad)} NaN or infinite "
                        f"reading{'s' if len(bad) != 1 else ''}, first at step {int(steps[bad[0]])}."
                        + (" Conflicting repeated-step values prevent trajectory analysis; "
                           "these non-finite readings remain observed." if ambiguous else "")
                    ),
                    evidence={"first_bad_step": int(steps[bad[0]]), "count": int(len(bad)),
                              "distinct_bad_steps": len(set(steps[bad])),
                              "trajectory_status": "conflicting_values" if ambiguous else "available"},
                    fixes=[
                        "Inspect gradient computation and norm measurement at these steps; a logged non-finite norm alone does not establish its cause.",
                        "If using mixed precision, inspect the gradient scaler and whether the optimizer skipped these updates.",
                        "If the loss went non-finite too, see the divergence finding.",
                    ],
                )
            )
        return findings

    def _explosions(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        groups: list[list[int]] = []
        ratio = np.zeros(len(values))
        for start, stop in finite_spans(values):
            segment = values[start:stop]
            if len(segment) < 5:
                continue
            z = robust_zscore(segment, self.window)
            median, _mad = rolling_median_mad(segment, self.window)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio[start:stop] = np.where(median > 0, segment / median, 0.0)
            flagged = [start + i for i in range(len(segment))
                       if z[i] > self.threshold and ratio[start + i] >= self.min_ratio]
            groups.extend(group_consecutive(flagged, gap=2))
        if not groups:
            return []
        flagged = [i for group in groups for i in group]
        peak_i = max(flagged, key=lambda i: ratio[i])
        return [
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
                    "median around it. Compare the logged loss and optimizer updates; "
                    "the norm alone does not establish clipping or an effect on training."
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
        ]

    def _saturation(self, steps: FloatArray, values: FloatArray) -> list[Finding]:
        n = len(values)
        if n < self.sat_window:
            return []
        window = values[-self.sat_window :]
        if not np.all(np.isfinite(window)) or np.any(window < 0):
            return []
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
                title="Gradient norms concentrated near their maximum",
                severity="info",
                metric=self.metric,
                step_start=int(steps[-self.sat_window]),
                step_end=int(steps[-1]),
                message=(
                    f"{fraction * 100:.0f}% of the last {self.sat_window} grad_norm readings "
                    f"sit within {self.sat_tol * 100:.1f}% of their observed maximum "
                    f"({ceiling:.4g}). Constant gradients and post-clip logging can both "
                    "produce this pattern. The norm history alone cannot establish "
                    "whether clipping was used, its threshold, or how often it changed updates."
                ),
                evidence={"ceiling": ceiling, "fraction_at_ceiling": fraction,
                          "window": self.sat_window, "clipping_status": "unknown"},
                fixes=[
                    "Check whether the norm is measured before or after clipping.",
                    "To measure clipping frequency, log the configured threshold and whether each update was clipped, or its before/after norms.",
                    "Compare clipping with the loss and parameter updates before changing its threshold.",
                ],
            )
        ]
