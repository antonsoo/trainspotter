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

    def finalize(self) -> None:
        """Sort every series by step. Readers append in encounter order;
        call this once after loading before anything reads the series."""
        for series in self.metrics.values():
            series.points.sort(key=lambda p: p.step)
