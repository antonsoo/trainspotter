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

An excursion shorter than the smoothing window is not a change of level and
has no share of the run worth reporting, but it can still be a burst: a
handful of logged points far above where the run was. It is reported as a
"Loss burst" `warning` when the level stood at least `burst_share` (default
half) of the run's progress above the earlier best and that rise is more
than `burst_threshold` (default 6.0) times the scatter of single points, on
the log scale for a positive series. Those are the two conditions `spikes`
sets for one point. A centred median only moves when more than half its
window is high, so such a burst is at least three logged points in a log
of under 150, and about 2% of the run in a longer one. A run with more than
`max_bursts` (default 5) gets one finding that counts them.

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

**Why bursts.** `spikes` scores a point against the centred median around
it, and a burst that fills more than half that window lifts the median with
it, so every point in it scores near zero. Until bursts were reported here,
three injected bursts in five went unreported at any height
(`studies/hub-census/`), and so did a public run whose loss stood at 6.8
against a level of 0.83 for 37 logged points.

**False-positive modes.**
- An epoch boundary can shift the level (data order, a harder shard) for a
  while without anything being wrong.
- A learning-rate restart raises the loss on purpose.
- The smoothing window spans 4% of the run (never fewer than five points). A
  rise shorter than half of it does not move the level and is left to
  `spikes`; one between half and the whole of it is a burst or nothing.
- The first points of a run scatter more than the rest, and the scatter a
  burst is judged against is the whole run's.
"""

from __future__ import annotations

import math

import numpy as np

from trainspotter.model import Run

from .base import (
    Detector,
    Excursion,
    Finding,
    Levels,
    excursions,
    levels,
    progress_text,
    series_arrays,
)


class SetbackDetector(Detector):
    name = "setbacks"

    def __init__(
        self,
        metric: str = "train/loss",
        info_share: float = 0.01,
        warn_share: float = 0.05,
        burst_share: float = 0.5,
        burst_threshold: float = 6.0,
        max_bursts: int = 5,
    ) -> None:
        self.metric = metric
        self.info_share = info_share
        self.warn_share = warn_share
        self.burst_share = burst_share
        self.burst_threshold = burst_threshold
        self.max_bursts = max_bursts

    def _unusual(self, lv: Levels, excursion: Excursion) -> bool:
        """Whether the level rose by more than `burst_threshold` times the scatter of single
        points, on the log scale for a positive series: the bar `spikes` sets for one point."""
        if lv.relative_scatter is not None and excursion.base > 0:
            score = math.log(excursion.peak / excursion.base) / max(lv.relative_scatter, 1e-3)
        else:
            score = (excursion.peak - excursion.base) / max(lv.scatter, 1e-12)
        return score > self.burst_threshold

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
        bursts: list[tuple[int, int, int, float, Excursion]] = []
        for excursion in excursions(lv):
            if not excursion.recovered:
                continue
            first, last = int(steps[excursion.start_index]), int(steps[excursion.end_index])
            share = (last - first) / span
            height = (excursion.peak - excursion.base) / lv.progress
            points = excursion.end_index - excursion.start_index
            if points + 1 < lv.window:
                # Shorter than the smoothing window is not a change of level. It is a burst.
                if height >= self.burst_share and self._unusual(lv, excursion):
                    bursts.append((first, last, points, height, excursion))
                continue
            if share < self.info_share:
                continue
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
        fixes = [
            "Look at what the run was fed between those steps: a bad shard, a change of data source, an epoch boundary.",
            "Lower the learning rate or clip gradients if the rise follows a large update.",
        ]
        if len(bursts) > self.max_bursts:
            first, last, points, height, worst = max(bursts, key=lambda burst: burst[3])
            findings.append(
                Finding(
                    detector=self.name,
                    title="Repeated loss bursts",
                    severity="warning",
                    metric=self.metric,
                    step_start=bursts[0][0],
                    step_end=bursts[-1][1],
                    message=(
                        f"{self.metric} rose far above a level it had already reached and came "
                        f"back {len(bursts)} times between step {bursts[0][0]} and step "
                        f"{bursts[-1][1]}, for up to {max(burst[2] for burst in bursts)} logged "
                        f"points at a time. The largest left a level of {worst.base:.4g} at step "
                        f"{first} and rose to {worst.peak:.4g}, {progress_text(height)}."
                    ),
                    evidence={
                        "count": len(bursts),
                        "longest_points": max(burst[2] for burst in bursts),
                        "peak_step": first,
                        "peak_level": worst.peak,
                        "share_of_progress": height,
                    },
                    fixes=fixes,
                )
            )
        else:
            for first, last, points, height, excursion in bursts:
                findings.append(
                    Finding(
                        detector=self.name,
                        title="Loss burst",
                        severity="warning",
                        metric=self.metric,
                        step_start=first,
                        step_end=last,
                        message=(
                            f"{self.metric} left a level of {excursion.base:.4g} at step {first}, "
                            f"rose to {excursion.peak:.4g}, and was back by step {last}: "
                            f"{points} logged point{'s' if points != 1 else ''}, "
                            f"{progress_text(height)} above a level it had already reached."
                        ),
                        evidence={
                            "level_before": excursion.base,
                            "peak_level": excursion.peak,
                            "points": points,
                            "steps": last - first,
                            "share_of_progress": height,
                        },
                        fixes=fixes,
                    )
                )
        return sorted(findings, key=lambda finding: finding.step_start)
