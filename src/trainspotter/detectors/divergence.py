"""Divergence detector.

**Algorithm.** Two independent triggers, either sufficient:

1. *Non-finite*: any logged value is NaN or +/-Inf. In the training loss
   that is an `error`: the logged loss is unusable. It does not establish
   whether optimizer updates failed or only the logging reduction did. In an
   evaluation metric it is a `warning`: 6 of the 1,639 public runs that
   ended at their best loss (`studies/hub-census/`) logged a NaN evaluation
   loss along the way.
2. *Did not recover* (`train/loss` only): reduce the final finite segment to its levels
   (`detectors.base.levels`: where it started, the best smoothed level it
   reached, the level of its last 5% of points). If the log ends inside an
   excursion above the best level (`detectors.base.excursions`), the share
   of the run's progress given back at the end sets the severity:
   `warning` from `warn_share` (default 25%), `error` from `error_share`
   (default 50%). The final level must also sit at least four times
   the level's noise above the best. A loss that never got clearly below
   its first level has no progress to measure against: it is an `error`
   when it ends at least `growth_ratio` (default 1.5) times its first level.

NaN/Inf readings separate the trajectory. The terminal-level check requires a
finite last observation; it never substitutes the last finite value for a
non-finite ending. Progress and growth baselines belong to the final segment,
and findings disclose that scope when the history contains a gap.

**Why levels.** The earlier test compared the last *point* with the lowest
*point* and asked for a rising tail. On noisy per-step losses the lowest
point is an outlier well below the level, so "1.5x the minimum" was easy to
reach in a healthy run. It also ran on `eval/loss`, where a rise beside a
training loss still at its best is overfitting and has a different remedy:
of the 57 public runs it called diverged for a rising tail
(`studies/hub-census/`), 26 were on `eval/loss`. A rising `eval/loss` is now left to
the `overfitting` detector.

**False-positive modes.**
- A metric that is supposed to increase will trip this for the right
  reason -- point it only at metrics where lower is better.
- A cyclic schedule or a curriculum that raises difficulty late produces a
  real rise that would have recovered; the log ending is all this sees.
- A run stopped during a temporary setback is reported as not recovered,
  which is true of the log and may not be true of the run.
"""

from __future__ import annotations

import math

from trainspotter.model import Run

from .base import (
    Detector,
    Finding,
    FloatArray,
    Levels,
    Severity,
    add_segment_evidence,
    excursions,
    finite_spans,
    levels,
    series_arrays,
)

DEFAULT_METRICS = ("train/loss", "eval/loss")


class DivergenceDetector(Detector):
    name = "divergence"

    def __init__(
        self,
        metrics: tuple[str, ...] = DEFAULT_METRICS,
        level_metric: str = "train/loss",
        warn_share: float = 0.25,
        error_share: float = 0.5,
        growth_ratio: float = 1.5,
    ) -> None:
        self.metrics = metrics
        self.level_metric = level_metric
        self.warn_share = warn_share
        self.error_share = error_share
        self.growth_ratio = growth_ratio

    def _run(self, run: Run) -> list[Finding]:
        findings: list[Finding] = []
        for metric_name in self.metrics:
            series = run.get(metric_name)
            if series is None or series.is_empty():
                # A conflict prevents a trajectory, not an observation that NaN/Inf
                # was recorded. Keep this pointwise fact without guessing a curve.
                excluded = run._excluded_metrics.get(metric_name)
                if excluded is not None:
                    bad = [point for point in excluded.points
                           if point.value is not None and not math.isfinite(point.value)]
                    if bad:
                        bad_steps = {point.step for point in bad}
                        findings.append(
                            Finding(
                                detector=self.name,
                                title="Non-finite readings in ambiguous input",
                                severity="error",
                                metric=metric_name,
                                step_start=min(bad_steps),
                                step_end=max(bad_steps),
                                message=(
                                    f"{metric_name} contains {len(bad)} non-finite readings at "
                                    f"{len(bad_steps)} distinct steps. Conflicting values elsewhere or at "
                                    "these steps prevent trajectory analysis; no reading was chosen as the winner."
                                ),
                                evidence={
                                    "first_bad_step": min(bad_steps),
                                    "reading_count": len(bad),
                                    "distinct_bad_steps": len(bad_steps),
                                },
                                fixes=[
                                    "Inspect repeated-step evidence and export one run/writer before interpreting the trajectory."
                                ],
                            )
                        )
                continue
            steps, values = series_arrays(series)
            observed = run.observed_series(metric_name)
            if observed is not None:
                observed_steps, observed_values = (
                    (steps, values) if observed is series else series_arrays(observed)
                )
                findings.extend(self._non_finite(metric_name, observed_steps, observed_values))
            if metric_name == self.level_metric:
                spans = finite_spans(values)
                # Terminal-level claims require a finite final observation. An
                # earlier level is not where a log ending in NaN/Inf ended.
                if spans and spans[-1][1] == len(values):
                    start, stop = spans[-1]
                    segment_steps = steps[start:stop]
                    segment = self._did_not_recover(
                        metric_name, segment_steps, values[start:stop],
                        scope="final finite segment" if start else "run",
                    )
                    if start:
                        add_segment_evidence(segment, segment_steps,
                                             unreadable=series.readable_count() != len(series))
                    findings.extend(segment)
        return findings

    def _non_finite(self, metric_name: str, steps: FloatArray, values: FloatArray) -> list[Finding]:
        bad = [i for i in range(len(values)) if not math.isfinite(values[i])]
        if not bad:
            return []
        first, last = bad[0], bad[-1]
        is_training = metric_name == self.level_metric
        kinds = sorted({"NaN" if math.isnan(values[i]) else "Inf" for i in bad})
        return [
            Finding(
                detector=self.name,
                title="Non-finite training loss" if is_training else "Non-finite evaluation values",
                # Keep the loss gate strict without inferring a training outcome
                # from a logging observation (e.g. mean over an empty tensor).
                severity="error" if is_training else "warning",
                metric=metric_name,
                step_start=int(steps[first]),
                step_end=int(steps[last]),
                message=(
                    f"{metric_name} became {'/'.join(kinds)} at step {int(steps[first])} "
                    f"({len(bad)} non-finite point(s) in this run)."
                ),
                evidence={"first_bad_step": int(steps[first]), "count": len(bad)},
                fixes=[
                    "Inspect the objective and logging reduction separately: an empty batch or empty logging window can produce NaN even while optimizer updates remain finite.",
                    "Check for a divide-by-zero or log(0) in the loss calculation.",
                    "If the optimizer's loss or gradients also become non-finite, inspect the learning rate, gradient clipping and mixed-precision loss scaling.",
                ],
            )
        ]

    def _did_not_recover(
        self, metric_name: str, steps: FloatArray, values: FloatArray, *, scope: str,
    ) -> list[Finding]:
        lv = levels(values)
        if lv is None:
            return []
        if not lv.has_progress:
            return self._rose_from_the_start(metric_name, steps, lv, scope=scope)
        found = excursions(lv)
        if not found or found[-1].recovered:
            return []
        last = found[-1]
        gave_back = lv.gave_back
        if gave_back < self.warn_share or lv.final - lv.best < 4.0 * lv.noise:
            return []
        severity: Severity = "error" if gave_back >= self.error_share else "warning"
        best_step = int(steps[lv.best_index])
        rise_step = int(steps[last.start_index])
        return [
            Finding(
                detector=self.name,
                title="Diverged (did not recover)",
                severity=severity,
                metric=metric_name,
                step_start=rise_step,
                step_end=int(steps[-1]),
                message=(
                    f"{metric_name} ended at {lv.final:.4g} after reaching {lv.best:.4g} around step "
                    f"{best_step}. It has been above that level since step {rise_step} and the log "
                    f"ends with {gave_back:.0%} of the {scope}'s progress given back."
                ),
                evidence={
                    "start_level": lv.start,
                    "best_level": lv.best,
                    "best_step": best_step,
                    "final_level": lv.final,
                    "gave_back": gave_back,
                    "rising_since_step": rise_step,
                },
                fixes=[
                    f"Use a checkpoint from around step {best_step}, before the rise.",
                    "Lower the learning rate or restart from the last good checkpoint before this window.",
                    "Check the LR schedule isn't re-warming up unintentionally (see lr_schedule detector).",
                    "Look for a data distribution shift at this step range (new shard, curriculum stage change).",
                ],
            )
        ]

    def _rose_from_the_start(
        self, metric_name: str, steps: FloatArray, lv: Levels, *, scope: str,
    ) -> list[Finding]:
        """A loss that never got clearly below its first level has no progress to give
        back. If it ended well above that first level, it only ever went up."""
        clear = max(6.0 * lv.noise, 4.0 * lv.start_noise)
        if lv.start <= 0 or lv.final < self.growth_ratio * lv.start or lv.final - lv.start < clear:
            return []
        return [
            Finding(
                detector=self.name,
                title="Diverged (rose from the start)" if scope == "run" else "Loss rose within final finite segment",
                severity="error",
                metric=metric_name,
                step_start=int(steps[0]),
                step_end=int(steps[-1]),
                message=(
                    ("" if scope == "run" else f"Within the {scope}, ")
                    + f"{metric_name} started around {lv.start:.4g} and ended at {lv.final:.4g}, "
                    f"{lv.final / lv.start:.1f}x higher, without ever settling below its first level."
                ),
                evidence={"start_level": lv.start, "final_level": lv.final, "ratio": lv.final / lv.start},
                fixes=[
                    "Lower the learning rate; a loss that climbs from the first step is usually an LR an order of magnitude too high.",
                    "Add LR warmup and gradient clipping.",
                    "Check the loss is the quantity being minimized (sign, reduction, label alignment).",
                ],
            )
        ]
