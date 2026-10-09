"""Describe repeated changes of direction in an evaluation history.

The public ``eval_noise`` detector ID is retained for compatibility, but an
aggregate curve cannot estimate measurement uncertainty: checkpoint quality,
sampling, and changes of evaluation data can all move the logged metric.

Within each contiguous finite segment of at least six observations, require
at least two reversals of the nonzero first differences. A monotone curve or
a single valley/peak is not fluctuation. Then compare the median absolute
adjacent change with the observed range, reporting a note from 15%. These
are descriptive, cadence-dependent thresholds, not a noise estimator or a
test of whether two checkpoints differ. Equal consecutive readings do not
introduce direction changes. NaN/Inf break the segment; never bridge them.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, finite_spans, series_arrays

DEFAULT_RATIO_THRESHOLD = 0.15


class EvalNoiseDetector(Detector):
    name = "eval_noise"

    def __init__(self, metric: str = "eval/loss", ratio_threshold: float = DEFAULT_RATIO_THRESHOLD, min_points: int = 6) -> None:
        self.metric = metric
        self.ratio_threshold = ratio_threshold
        self.min_points = min_points

    def _run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None or len(series) < self.min_points:
            return []
        steps, values = series_arrays(series)
        findings = []
        for start, stop in finite_spans(values):
            segment = values[start:stop]
            if len(segment) < max(3, self.min_points):
                continue
            # Normalize before subtracting: finite loss values can span most
            # of float64's range. The ratio is invariant to this scaling.
            scale = float(np.max(np.abs(segment)))
            if scale == 0:
                continue
            scaled = segment / scale
            diffs = np.diff(scaled)
            directions = np.sign(diffs[diffs != 0])
            reversals = int(np.count_nonzero(np.diff(directions)))
            if reversals < 2:
                continue
            change = float(np.median(np.abs(diffs)))
            value_range = float(np.max(scaled) - np.min(scaled))
            ratio = change / value_range
            if ratio < self.ratio_threshold:
                continue
            # Raw changes/ranges may overflow even with finite observations.
            findings.append(
                Finding(
                    detector=self.name,
                    title="Evaluation values fluctuate",
                    severity="info",
                    metric=self.metric,
                    step_start=int(steps[start]),
                    step_end=int(steps[stop - 1]),
                    message=(
                        f"{self.metric} changes direction {reversals} times across "
                        f"{len(segment)} finite observations. Its median absolute adjacent "
                        f"change is {ratio * 100:.0f}% of the observed range. "
                        "This describes the logged curve; it does not measure evaluation "
                        "uncertainty or establish whether checkpoints differ."
                    ),
                    evidence={
                        "observations": len(segment),
                        "direction_changes": reversals,
                        "upward_intervals": int(np.count_nonzero(diffs > 0)),
                        "downward_intervals": int(np.count_nonzero(diffs < 0)),
                        "scale": scale,
                        "scaled_median_absolute_change": change,
                        "scaled_value_range": value_range,
                        "ratio": ratio,
                    },
                    fixes=[
                        "Check whether evaluation data, sampling, or metric definitions changed between logged values.",
                        "To compare saved checkpoints, evaluate them on the same examples and retain paired per-example scores; account for dependent examples.",
                        "Repeated evaluation only measures randomness present in that procedure; repeating a deterministic pass does not measure uncertainty across examples.",
                    ],
                )
            )
        return findings
