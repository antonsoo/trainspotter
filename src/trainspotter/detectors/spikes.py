"""Loss-spike detector.

**Algorithm.** Compute a trailing rolling median and MAD (median absolute
deviation) of the target metric with window `window` (default 21 steps),
then a modified z-score `0.6745 * (x - median) / MAD` at every step (see
`detectors.base.robust_zscore`). Flag any step whose |z| exceeds
`threshold` (default 6.0). Robust statistics are used instead of a rolling
mean/stddev because a spike would otherwise poison the very baseline used
to judge it -- median and MAD barely move when one point is a huge outlier.
Consecutive flagged steps are merged into a single finding spanning their
step range, reported against the metric's peak value in that range.

**False-positive modes.**
- A genuinely bumpy metric (small `window`, high natural variance) inflates
  the z-score of ordinary noise; widen `window` or raise `threshold`.
- A deliberate LR-schedule restart or curriculum/dataset-phase change can
  cause a real, intended jump that looks identical to a spike.
- Very short runs (fewer than ~2*window steps) give MAD too little history
  to be a stable denominator, so early findings are less trustworthy.
"""

from __future__ import annotations

import math

from trainspotter.model import Run

from .base import Detector, Finding, group_consecutive, robust_zscore, series_arrays

DEFAULT_WINDOW = 21
DEFAULT_THRESHOLD = 6.0


class SpikeDetector(Detector):
    name = "spikes"

    def __init__(
        self,
        metrics: tuple[str, ...] = ("train/loss",),
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.metrics = metrics
        self.window = window
        self.threshold = threshold

    def run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None or len(series) < 5:
                continue
            steps, values = series_arrays(series)
            z = robust_zscore(values, self.window)
            flagged = [i for i in range(len(values)) if abs(z[i]) > self.threshold]
            for group in group_consecutive(flagged, gap=2):
                peak_i = max(group, key=lambda i: abs(z[i]))
                peak_z = abs(z[peak_i])
                scale_note = (
                    "far beyond the local median-absolute-deviation scale (it's non-finite)"
                    if not math.isfinite(peak_z)
                    else f"{peak_z:.1f}x the local median-absolute-deviation scale"
                )
                findings.append(
                    Finding(
                        detector=self.name,
                        title="Loss spike",
                        severity="warning" if peak_z < self.threshold * 1.8 else "error",
                        metric=metric_name,
                        step_start=int(steps[group[0]]),
                        step_end=int(steps[group[-1]]),
                        message=(
                            f"{metric_name} jumped to {values[peak_i]:.4g} at step "
                            f"{int(steps[peak_i])}, {scale_note}."
                        ),
                        evidence={
                            "peak_step": int(steps[peak_i]),
                            "peak_value": float(values[peak_i]),
                            "robust_z": float(z[peak_i]),
                            "window": self.window,
                            "threshold": self.threshold,
                        },
                        fixes=[
                            "Lower the learning rate or add/extend LR warmup.",
                            "Clip gradients (or lower the existing max-norm).",
                            "Check the batch at this step for a data or tokenization bug (e.g. an unmasked pad token, a corrupted shard).",
                            "If it self-recovers within a few steps and doesn't recur, it may be benign -- confirm against the divergence detector's verdict.",
                        ],
                    )
                )
        return findings
