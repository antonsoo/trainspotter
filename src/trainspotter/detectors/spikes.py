"""Loss-spike detector.

**Algorithm.** A point is a spike when it is both unusual and large:

1. *Unusual*: it sits more than `threshold` (default 6.0) robust standard
   deviations above the level around it. The score is taken on the
   logarithm of the loss when every value is positive, because the scatter
   of a mini-batch loss is multiplicative and heavy-tailed: a point at
   three times the level is ordinary in a run whose points routinely
   double, and glaring in a smooth one. The scale is the larger of the
   run-wide scatter and that of the trailing `window` (default 21) points,
   both from the median absolute deviation, so a spike cannot inflate the
   scale it is judged against and a quiet window cannot shrink it.
2. *Large*: it stands above the level around it (`detectors.base.levels`)
   by a share of the run's whole progress, the distance the level fell from
   its start to its best. From `warn_share` (default half) it is a
   `warning`, reported with that share. Between `info_share` (default a
   tenth) and that, all such points become one `info` finding that counts
   them. Below a tenth nothing is reported.

Height is measured from the level around the point or from the median of
the five points before it, whichever is higher, and the first three points
of a run are never spikes: a run starts high and comes down, and that
descent is not an event. Consecutive flagged steps are merged. A run with more than `max_findings`
(default 5) warnings gets one finding that counts them.

**Why both.** On 3,145 public `Trainer` logs (`studies/hub-census/`) the
first condition alone, on the raw loss with a 21-point scale, fired on 37%
of runs, 17 times each on average, and two thirds of those runs ended at
their best loss. A jump from 0.004 to 0.007 in a run that came down from
0.07 is such a point. The second condition asks whether the jump was big next to
what the run had achieved, which is what a reader means by a spike.
Downward points are never reported.

**False-positive modes.**
- A deliberate LR restart or a change of dataset phase causes a real,
  intended jump that looks the same.
- With fewer than about `2 * window` points the MAD has little history and
  early findings are less trustworthy.
- A metric that never falls (a reward-driven objective logged as "loss")
  has no progress to measure against, and is not examined.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import (
    Detector,
    Finding,
    FloatArray,
    Levels,
    group_consecutive,
    levels,
    progress_text,
    rolling_median_mad,
    series_arrays,
)

DEFAULT_WINDOW = 21
DEFAULT_THRESHOLD = 6.0
#: A point needs this many earlier points to be judged against.
HISTORY = 3


class SpikeDetector(Detector):
    name = "spikes"

    def __init__(
        self,
        metrics: tuple[str, ...] = ("train/loss",),
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
        info_share: float = 0.10,
        warn_share: float = 0.5,
        max_findings: int = 5,
    ) -> None:
        self.metrics = metrics
        self.window = window
        self.threshold = threshold
        self.info_share = info_share
        self.warn_share = warn_share
        self.max_findings = max_findings

    def _scores(self, values: FloatArray, lv: Levels) -> FloatArray:
        """How many robust standard deviations each point sits above the level around it.

        The scale is the larger of the run-wide scatter and the scatter of the trailing
        window. A 21-point window alone underestimates it often enough that, over
        thousands of points, ordinary ones score above any threshold.
        """
        if lv.relative_scatter is not None:
            _median, mad = rolling_median_mad(np.log(values), self.window)
            # At least 0.1% scatter, so a noiseless stretch does not turn its next
            # ordinary step into an infinite score.
            scale = np.maximum(1.4826 * mad, max(lv.relative_scatter, 1e-3))
            scores: FloatArray = np.log(values / lv.smooth) / scale
            return scores
        _median, mad = rolling_median_mad(values, self.window)
        scale = np.maximum(1.4826 * mad, max(lv.scatter, 1e-12))
        linear: FloatArray = (values - lv.smooth) / scale
        return linear

    def _run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None or len(series) < 5:
                continue
            steps, values = series_arrays(series)
            finite = np.isfinite(values)
            # Non-finite readings are the divergence detector's to report.
            steps, values = steps[finite], values[finite]
            lv = levels(values)
            if lv is None or not lv.has_progress:
                continue
            z = self._scores(values, lv)
            # Height above the level around the point and above the points just before
            # it: the first readings of a run are high because training had not begun,
            # and the way down from them is not a spike.
            before = np.array(
                [np.median(values[max(0, i - 5) : i]) if i >= HISTORY else np.inf for i in range(len(values))]
            )
            share = (values - np.maximum(lv.smooth, before)) / lv.progress
            unusual = [i for i in range(HISTORY, len(values)) if z[i] > self.threshold]
            large = group_consecutive([i for i in unusual if share[i] >= self.warn_share], gap=2)
            modest = group_consecutive(
                [i for i in unusual if self.info_share <= share[i] < self.warn_share], gap=2
            )
            fixes = [
                "Lower the learning rate or add/extend LR warmup.",
                "Clip gradients (or lower the existing max-norm).",
                "Check the batch at this step for a data or tokenization bug (e.g. an unmasked pad token, a corrupted shard).",
            ]
            if len(large) > self.max_findings:
                worst = max((i for group in large for i in group), key=lambda i: share[i])
                findings.append(
                    Finding(
                        detector=self.name,
                        title="Repeated loss spikes",
                        severity="warning",
                        metric=metric_name,
                        step_start=int(steps[large[0][0]]),
                        step_end=int(steps[large[-1][-1]]),
                        message=(
                            f"{metric_name} spiked {len(large)} times between step "
                            f"{int(steps[large[0][0]])} and {int(steps[large[-1][-1]])}. The largest "
                            f"reached {values[worst]:.4g} at step {int(steps[worst])}, "
                            f"{progress_text(share[worst])} above the level around it "
                            f"({lv.smooth[worst]:.4g})."
                        ),
                        evidence={
                            "count": len(large),
                            "peak_step": int(steps[worst]),
                            "peak_value": float(values[worst]),
                            "share_of_progress": float(share[worst]),
                        },
                        fixes=fixes,
                    )
                )
            else:
                for group in large:
                    peak_i = max(group, key=lambda i: share[i])
                    findings.append(
                        Finding(
                            detector=self.name,
                            title="Loss spike",
                            severity="warning",
                            metric=metric_name,
                            step_start=int(steps[group[0]]),
                            step_end=int(steps[group[-1]]),
                            message=(
                                f"{metric_name} jumped to {values[peak_i]:.4g} at step "
                                f"{int(steps[peak_i])} from a level of {lv.smooth[peak_i]:.4g}, "
                                f"{progress_text(share[peak_i])}, for "
                                f"{len(group)} logged point{'s' if len(group) != 1 else ''}."
                            ),
                            evidence={
                                "peak_step": int(steps[peak_i]),
                                "peak_value": float(values[peak_i]),
                                "level": float(lv.smooth[peak_i]),
                                "share_of_progress": float(share[peak_i]),
                                "robust_z": float(z[peak_i]),
                                "window": self.window,
                                "threshold": self.threshold,
                            },
                            fixes=[
                                *fixes,
                                "If the level came straight back and it doesn't recur, it may be one bad batch.",
                            ],
                        )
                    )
            if modest:
                worst = max((i for group in modest for i in group), key=lambda i: share[i])
                findings.append(
                    Finding(
                        detector=self.name,
                        title="Isolated loss spikes",
                        severity="info",
                        metric=metric_name,
                        step_start=int(steps[modest[0][0]]),
                        step_end=int(steps[modest[-1][-1]]),
                        message=(
                            f"{metric_name} had {len(modest)} isolated point"
                            f"{'s' if len(modest) != 1 else ''} well above the level around "
                            f"{'them' if len(modest) != 1 else 'it'}; the largest was "
                            f"{values[worst]:.4g} at step {int(steps[worst])} against a level of "
                            f"{lv.smooth[worst]:.4g}, {progress_text(share[worst])}."
                        ),
                        evidence={
                            "count": len(modest),
                            "peak_step": int(steps[worst]),
                            "peak_value": float(values[worst]),
                            "share_of_progress": float(share[worst]),
                        },
                        fixes=["Usually one hard or malformed batch. Look at the data at that step if it recurs."],
                    )
                )
        return findings
