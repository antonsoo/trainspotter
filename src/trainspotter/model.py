"""The run-independent data model every reader converges on.

A training run, whatever produced it, is a bag of named scalar series
indexed by step. Everything downstream (detectors, reports) only ever
talks to this shape, never to HF/W&B/Lightning/TensorBoard specifics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Point:
    """One logged value of one metric."""

    step: int
    value: float
    wall_time: float | None = None
    epoch: float | None = None


@dataclass(slots=True)
class MetricSeries:
    """All logged points for a single metric name, kept in step order."""

    name: str
    points: list[Point] = field(default_factory=list)

    def steps(self) -> list[int]:
        return [p.step for p in self.points]

    def values(self) -> list[float]:
        return [p.value for p in self.points]

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0


@dataclass(slots=True)
class Run:
    """A whole training run: named metric series plus light metadata."""

    metrics: dict[str, MetricSeries] = field(default_factory=dict)
    source_format: str = "unknown"
    source_path: str = ""
    meta: dict[str, object] = field(default_factory=dict)

    def add_point(
        self, metric: str, step: int, value: float, wall_time: float | None = None
    ) -> None:
        series = self.metrics.setdefault(metric, MetricSeries(name=metric))
        series.points.append(Point(step=step, value=value, wall_time=wall_time))

    def get(self, name: str) -> MetricSeries | None:
        return self.metrics.get(name)

    def metric_names(self) -> list[str]:
        return sorted(self.metrics)

    def skipped_rows(self) -> int:
        """How many rows, lines or entries the reader could not use (unparsable JSON,
        no step, a step that isn't a number). A report built on part of a log says so."""
        keys = ("skipped_rows", "skipped_lines", "skipped_entries")
        return sum(n for n in (self.meta.get(key) for key in keys) if isinstance(n, int))

    def other_runs(self) -> tuple[str, list[str]] | None:
        """For a source that held several runs (a W&B chart export, a TensorBoard logdir):
        the name of the run that was read, and the names of the others. None for a source
        of one run. A report on one run out of several says which, and that there are more."""
        selected = self.meta.get("selected_run")
        runs = self.meta.get("runs")
        if not isinstance(selected, str) or not isinstance(runs, list):
            return None
        names = [r["name"] for r in runs if isinstance(r, dict) and isinstance(r.get("name"), str)]
        return selected, [name for name in names if name != selected]

    def overwritten_points(self) -> int:
        """Points a reader dropped because a later restart of the run wrote the same steps again."""
        n = self.meta.get("overwritten_points")
        return n if isinstance(n, int) else 0

    def finalize(self) -> None:
        """Sort every series by step. Readers append in encounter order;
        call this once after loading before anything reads the series."""
        for series in self.metrics.values():
            series.points.sort(key=lambda p: p.step)
