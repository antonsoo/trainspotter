"""The run-independent data model every reader converges on.

A training run, whatever produced it, is a bag of named scalar series
indexed by step. Everything downstream (detectors, reports) only ever
talks to this shape, never to HF/W&B/Lightning/TensorBoard specifics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

AxisKind = Literal["step", "logging_step", "row_index"]


@dataclass(frozen=True, slots=True)
class Point:
    """One recorded position; None means an unreadable value, not an observed NaN."""

    step: int
    value: float | None
    wall_time: float | None = None
    epoch: float | None = None


@dataclass(slots=True)
class RepeatedStep:
    """Readings sharing a metric and step; no guess about which writer is right."""

    step: int
    readings: list[Point]
    conflicting_values: bool
    conflicting_times: bool


@dataclass(slots=True)
class MetricSeries:
    """All logged points for a single metric name, kept in step order."""

    name: str
    points: list[Point] = field(default_factory=list)

    def steps(self) -> list[int]:
        return [p.step for p in self.points]

    def values(self) -> list[float]:
        """Numerical projection: unreadable positions break finite segments.

        Inspect points for original values. A projected NaN at an unreadable
        position is a gap sentinel, never evidence that NaN was logged.
        """
        return [p.value if p.value is not None else math.nan for p in self.points]

    def readable_count(self) -> int:
        return sum(p.value is not None for p in self.points)

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def analysis(self) -> tuple[list[Point], list[RepeatedStep]]:
        """One observation per step, or no trajectory for conflicting values.

        The original points remain untouched. Repeated equal values count once;
        unequal timestamps cannot provide a unique wall time for that step.
        Nonfinite *observed* values remain available to divergence detection.
        """
        if all(self.points[i - 1].step < point.step for i, point in enumerate(self.points) if i):
            return self.points, []
        groups: dict[int, list[Point]] = {}
        for point in self.points:
            groups.setdefault(point.step, []).append(point)
        points: list[Point] = []
        repeated: list[RepeatedStep] = []
        for step, readings in sorted(groups.items()):
            first = readings[0]
            observed = [p for p in readings if p.value is not None]
            same_values = all(
                p.value == observed[0].value
                or (p.value is not None and observed[0].value is not None
                    and math.isnan(p.value) and math.isnan(observed[0].value))
                for p in observed
            )
            same_times = all(p.wall_time == first.wall_time for p in readings)
            if len(readings) > 1:
                repeated.append(RepeatedStep(step, readings, not same_values, not same_times))
            points.append(
                Point(step, first.value if len(observed) == len(readings) else None,
                      first.wall_time if same_times else None, first.epoch)
            )
        return ([] if any(group.conflicting_values for group in repeated) else points), repeated


@dataclass(slots=True)
class Run:
    """A whole training run: named metric series plus light metadata."""

    metrics: dict[str, MetricSeries] = field(default_factory=dict)
    source_format: str = "unknown"
    source_path: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    _analysis_ready: bool = field(default=False, repr=False)
    _excluded_metrics: dict[str, MetricSeries] = field(default_factory=dict, repr=False)
    _original_metrics: dict[str, MetricSeries] = field(default_factory=dict, repr=False)
    _pending_unreadable: dict[str, list[Point]] = field(default_factory=dict, repr=False)

    def add_point(
        self, metric: str, step: int, value: float | None, wall_time: float | None = None
    ) -> None:
        series = self.metrics.setdefault(metric, MetricSeries(name=metric))
        series.points.extend(self._pending_unreadable.pop(metric, []))
        series.points.append(Point(step=step, value=value, wall_time=wall_time))
        self._analysis_ready = False

    def add_unreadable(
        self, metric: str, step: int, wall_time: float | None = None, *, known_metric: bool = False,
    ) -> None:
        """Keep a known coordinate without inventing a measurement.

        Text-only columns remain metadata. If a later numeric reading establishes
        that this is a metric, its earlier unreadable positions are retained too.
        """
        if known_metric or metric in self.metrics:
            self.add_point(metric, step, None, wall_time)
        else:
            self._pending_unreadable.setdefault(metric, []).append(Point(step, None, wall_time))
            self._analysis_ready = False

    def observed_series(self, name: str) -> MetricSeries | None:
        """Actual numeric readings, including NaN/Inf, before gap projection.

        Equal repeats count once; conflicting observations stay available for
        pointwise evidence even though they cannot establish a trajectory.
        """
        current = self.metrics.get(name)
        if current is not None and all(p.value is not None for p in current.points):
            return current
        series = self._original_metrics.get(name, current)
        if series is None:
            return None
        observed = MetricSeries(name, [p for p in series.points if p.value is not None])
        points, _repeated = observed.analysis()
        return MetricSeries(name, points) if points else observed

    def get(self, name: str) -> MetricSeries | None:
        return self.metrics.get(name)

    def metric_names(self) -> list[str]:
        return sorted(self.metrics)

    def axis_kind(self) -> AxisKind:
        """Coordinates can locate a reading without measuring training progress."""
        if self.meta.get("step_axis") == "logging_step":
            return "logging_step"
        column = self.meta.get("step_column")
        if isinstance(column, str) and column.startswith("(row index"):
            return "row_index"
        return "step"

    def axis_label(self) -> str:
        return {"step": "steps", "logging_step": "logging steps", "row_index": "row indices"}[
            self.axis_kind()
        ]

    def repeated_steps(self) -> dict[str, list[RepeatedStep]]:
        return {
            name: groups
            for name, series in self.metrics.items()
            if (groups := series.analysis()[1])
        }

    def analysis_view(self) -> Run:
        """A detector-only projection; reports retain the original readings."""
        if self._analysis_ready:
            return self
        metrics = {}
        excluded = {}
        for name, series in self.metrics.items():
            points, _repeated = series.analysis()
            if points:
                metrics[name] = MetricSeries(name, points)
            elif series.points:
                excluded[name] = series
        return Run(
            metrics=metrics,
            source_format=self.source_format,
            source_path=self.source_path,
            meta=self.meta,
            _analysis_ready=True,
            _excluded_metrics=excluded,
            _original_metrics=self.metrics,
        )

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
        return sum(
            count
            for name, count in invalid.items()
            if isinstance(count, int) and (name in self.metrics or name in known)
        )

    def input_complete(self) -> bool:
        """Supported input was read with unambiguous sources; not a health/coverage claim."""
        return (
            bool(self.metrics)
            and self.skipped_rows() == 0
            and self.skipped_values() == 0
            and self.unread_event_bytes() == 0
            and not self.meta.get("ambiguous_metric_sources")
            and all(p.value is not None for series in self.metrics.values() for p in series.points)
            and not any(
                group.conflicting_values or group.conflicting_times
                for groups in self.repeated_steps().values()
                for group in groups
            )
        )

    def input_notes(self) -> list[str]:
        notes = []
        for name, series in sorted(self.metrics.items()):
            unreadable = len(series) - series.readable_count()
            if unreadable:
                notes.append(
                    f"{name}: {unreadable} unreadable values at recorded steps; "
                    "trajectory checks stop at these gaps. These are not observed NaN/Inf values."
                )
        for name, groups in sorted(self.repeated_steps().items()):
            extra = sum(len(group.readings) - 1 for group in groups)
            conflicts = sum(group.conflicting_values for group in groups)
            timing = sum(group.conflicting_times for group in groups)
            if name == "epoch" and not conflicts and not timing:
                # `Trainer` writes the epoch into both the training entry and the evaluation
                # entry of a step. Two equal readings of where training was are one fact, and
                # nearly every real log with an evaluation has them.
                continue
            notes.append(
                f"{name}: {extra} repeated {'reading' if extra == 1 else 'readings'} "
                f"across {len(groups)} {'step' if len(groups) == 1 else 'steps'}; "
                "equal values count once in analysis, and all readings remain in evidence."
            )
            if conflicts:
                notes.append(
                    f"{name}: {conflicts} {'step has' if conflicts == 1 else 'steps have'} "
                    "conflicting values. This metric is "
                    "excluded from trajectory checks; separate the runs or writers first."
                )
            elif timing:
                notes.append(
                    f"{name}: {timing} repeated steps have different or missing wall times; "
                    "their timing is unknown. Value-only checks can still run."
                )
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
        if self.axis_kind() == "logging_step":
            notes.append(
                f"The horizontal axis uses W&B logging counter {axis!r}; "
                "findings locate logging events, not training steps. Log a training counter "
                "or select one with --step-key to measure progress per training step."
            )
        if self.meta.get("step_selection") == "explicit":
            notes.append(
                f"Training-step column {axis!r} was selected explicitly (--step-key); "
                "its values are interpreted as training-step counts."
            )
        if not self.input_complete():
            notes.append("Input is incomplete; findings describe only the readable data.")
        return notes

    def finalize(self) -> None:
        """Sort every series by step. Readers append in encounter order;
        call this once after loading before anything reads the series."""
        for series in self.metrics.values():
            series.points.sort(key=lambda p: p.step)
        self._pending_unreadable.clear()  # Columns with no numeric/known metric are text metadata.
