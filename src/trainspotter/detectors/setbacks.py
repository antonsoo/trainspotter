"""Setback detector: the loss level rose, stayed up, and came back.

**Algorithm.** Reduce `train/loss` to its smoothed level
(`detectors.base.levels`) and find every excursion: a stretch where the
level sat above the best it had already reached by a quarter of the run's
progress (and never less than four times the level's noise), ending when
the level was back near that earlier best (`detectors.base.excursions`). Each one
that was recovered from is reported with what it cost: the steps between
leaving the earlier level and returning to it, as a share of the run.
Below `info_share` (default 1%) nothing is reported; from `warn_share`
(default 5%) it is a `warning`.

An excursion the log ends inside is not reported here: the `divergence`
detector judges those by how much of the run's progress is gone at the end.

An excursion that begins in the first tenth of the run is titled "Early
rise". It is the commonest kind in public logs: a run that starts from a
trained checkpoint at a low loss, climbs while the learning rate warms up,
and spends a large part of its steps getting back to where it began.

**Why it exists.** A single noisy point is not what costs a run; a level
that climbs and takes thousands of steps to return is. The spike detector
looks at points and cannot see the second kind: a public run whose level
rose from 0.51 to 1.43 and spent 13,000 of its 39,000 steps above where it
had been got four spike findings from the earlier point detector and
nothing about the rise.

**False-positive modes.**
- An epoch boundary can shift the level (data order, a harder shard) for a
  while without anything being wrong.
- A learning-rate restart raises the loss on purpose.
- The smoothing window spans 4% of the run, so a setback shorter than that
  is not seen as a change of level.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, excursions, levels, series_arrays


class SetbackDetector(Detector):
    name = "setbacks"

    def __init__(
        self,
        metric: str = "train/loss",
        info_share: float = 0.01,
        warn_share: float = 0.05,
    ) -> None:
        self.metric = metric
        self.info_share = info_share
        self.warn_share = warn_share

    def _run(self, run: Run) -> list[Finding]:
        series = run.get(self.metric)
        if series is None:
            return []
        steps, values = series_arrays(series)
        finite = np.isfinite(values)
        steps, values = steps[finite], values[finite]
        lv = levels(values)
        if lv is None or not lv.has_progress:
            return []
        span = float(steps[-1] - steps[0])
        if span <= 0:
            return []
        findings: list[Finding] = []
        for excursion in excursions(lv):
            if not excursion.recovered:
                continue
            first, last = int(steps[excursion.start_index]), int(steps[excursion.end_index])
            share = (last - first) / span
            # Shorter than the smoothing window is not a change of level.
            if share < self.info_share or excursion.end_index - excursion.start_index + 1 < lv.window:
                continue
            height = (excursion.peak - excursion.base) / lv.progress
            early = excursion.start_index < len(values) / 10
            findings.append(
                Finding(
                    detector=self.name,
                    title="Early rise" if early else "Setback",
                    severity="warning" if share >= self.warn_share else "info",
                    metric=self.metric,
                    step_start=first,
                    step_end=last,
                    message=(
                        f"{self.metric} left a level of {excursion.base:.4g} at step {first}, rose to "
                        f"{excursion.peak:.4g}, and was back by step {last}: {last - first} steps "
                        f"({share:.0%} of the run) above a level it had already reached, with "
                        f"{height:.0%} of the run's progress given back at the worst point."
                    ),
                    evidence={
                        "level_before": excursion.base,
                        "peak_level": excursion.peak,
                        "steps_lost": last - first,
                        "share_of_run": share,
                        "share_of_progress": height,
                    },
                    fixes=[
                        "If training resumed from a checkpoint, restore the optimizer and scheduler state with it, or start from a lower learning rate.",
                        "Lengthen the warmup or lower the peak learning rate.",
                    ]
                    if early
                    else [
                        "Look at what changed at the first step: the learning rate, a new data shard, an epoch boundary.",
                        "Lower the peak learning rate or clip gradients if the rise follows a large update.",
                        "If this was a planned restart or curriculum stage, it is expected.",
                    ],
                )
            )
        return findings
