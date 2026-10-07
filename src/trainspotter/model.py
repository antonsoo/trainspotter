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
        keys = ("skipped_rows", "skipped_lines", "skipped_entries", "skipped_events")
        return sum(max(0, n) for n in (self.meta.get(key) for key in keys) if type(n) is int)

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
        return max(0, n) if type(n) is int else 0

    def unread_event_bytes(self) -> int:
        """Bytes TensorBoard could not decode, including a partially written last record."""
        n = self.meta.get("unread_event_bytes", 0)
        return max(0, n) if type(n) is int else 0

    def skipped_values(self) -> int:
        """Invalid nonblank cells in a recognized metric or a column with numeric readings."""
        invalid = self.meta.get("invalid_metric_values", {})
        known = self.meta.get("invalid_known_metrics", [])
        if not isinstance(invalid, dict) or not isinstance(known, list):
            return 0
        return sum(count for name, count in invalid.items()
                   if isinstance(count, int) and (name in self.metrics or name in known))

    def input_complete(self) -> bool:
        """Supported input was read with unambiguous sources; not a health/coverage claim."""
        return (bool(self.metrics) and self.skipped_rows() == 0
                and self.skipped_values() == 0 and self.unread_event_bytes() == 0
                and not self.meta.get("ambiguous_metric_sources"))

    def input_notes(self) -> list[str]:
        notes = []
        if self.skipped_rows():
            notes.append(f"{self.skipped_rows()} rows or events could not be read.")
        if self.skipped_values():
            notes.append(f"{self.skipped_values()} nonblank metric values were not numeric.")
        if self.unread_event_bytes():
            notes.append(
                f"{self.unread_event_bytes()} event-file bytes could not be decoded "
                "(a partial write or corrupt record)."
            )
        ambiguous = self.meta.get("ambiguous_metric_sources")
        if isinstance(ambiguous, list) and ambiguous:
            notes.append(
                f"{len(ambiguous)} metric names are claimed by multiple writer directories; "
                "their curves are kept separate. Select a single writer directory to diagnose them."
            )
        axis = self.meta.get("step_column")
        if isinstance(axis, str) and axis.startswith("(row index"):
            notes.append(f"The horizontal axis uses {axis}; these are not training-step counts.")
        if not self.input_complete():
            notes.append("Input is incomplete; findings describe only the readable data.")
        return notes

    def finalize(self) -> None:
        """Sort every series by step. Readers append in encounter order;
        call this once after loading before anything reads the series."""
        for series in self.metrics.values():
            series.points.sort(key=lambda p: p.step)
