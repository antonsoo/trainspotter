"""Weights & Biases CSV readers: a run's history, and a chart's export.

**Run history.** From the public API, `wandb.Api().run(path).history()`
written to CSV: columns `_step`, `_runtime`, `_timestamp`, and one column per
logged metric, named the way you called `wandb.log` (e.g. `train/loss`,
`eval/loss`).

`_step` counts committed logging events. A single recognized training counter
(`trainer/global_step`, `train/global_step`, `global_step`, or a separate `step`
column) supplies training coordinates instead. `step_key` selects an exact
column explicitly. Without one, `_step` stays a disclosed logging axis and
cannot support wall-clock seconds per training step. See docs/wandb-axes.md.

**Chart export.** The file a chart's "Export" menu downloads
(`wandb_export_<time>.csv`): an x-axis column (`Step`, unless the chart's axis
was changed) followed, for every run drawn in the chart, by three columns
per metric: `<run name> - <metric>`, `<run name> - <metric>__MIN` and
`<run name> - <metric>__MAX`. The MIN and MAX columns are the band the chart
draws around a smoothed or grouped line and are not read. A chart with
several runs is several runs in one file: one of them is analysed, and the
others are listed (see `select_run`), because detectors compare the metrics
of a run with each other, and a run's train loss says nothing about another
run's eval loss.
The default `Step` axis is also a logging counter. An explicit `step_key` must
name the first column: each chart export has only one shared coordinate axis.
Multiple ` - ` separators do not identify the run/metric boundary. Such labels
require `chart_single_run`, an explicit assertion that the entire chart belongs
to one exact display name. See docs/wandb-chart-identity.md for the limitations.

A third export, the runs table (one row per run: `Name`, `State`,
`Runtime`, the config and summary columns), is not a training log at all;
`looks_like_runs_table` lets the caller say so and stop before rows of
different runs are read as consecutive steps of one.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from trainspotter.model import Run

from .common import (
    metric_names,
    metric_value,
    select_run,
    try_float,
    try_step,
    unreadable_row,
    validate_columns,
)
from .tabular import _decimal_point, _decimal_points, _delimiter, _find_step_key

_STEP_COL = "_step"
_RUNTIME_COL = "_runtime"
_SKIP_PREFIXES = ("_",)
_BAND_SUFFIXES = ("__MIN", "__MAX")
# W&B's default Step/_step counts log commits, not optimizer updates. Only
# separately logged counters (or an explicit caller selection) establish that axis.
_TRAINING_AXES = {"global_step", "trainer/global_step", "train/global_step"}
_LOGGING_AXES = {"step", "_step"}
_RUNS_TABLE_COLUMNS = {
    "State",
    "Runtime",
    "Created",
    "Sweep",
    "User",
    "Tags",
    "Notes",
    "ID",
    "Updated",
    "End Time",
}


def is_chart_export(fieldnames: Sequence[str]) -> bool:
    """A chart export is recognized by its bands: a column with `__MIN` and `__MAX` companions."""
    names = set(fieldnames)
    bases = [name[: -len("__MIN")] for name in fieldnames if name.endswith("__MIN")]
    return any(base in names and base + "__MAX" in names for base in bases)


def looks_like_runs_table(fieldnames: Sequence[str]) -> bool:
    """The runs table: a `Name` column first, the table's own columns, and no step."""
    if not fieldnames or fieldnames[0] != "Name":
        return False
    has_step = bool({"step", "_step", "global_step"} & {name.lower() for name in fieldnames})
    return bool(_RUNS_TABLE_COLUMNS & set(fieldnames)) and not has_step


def _split_run(column: str, single_run: str | None) -> tuple[str, str]:
    """A label has no escaping that distinguishes a separator inside either name."""
    if single_run is not None:
        prefix = single_run + " - "
        if not column.startswith(prefix):
            raise ValueError(
                f"chart column {column!r} does not start with {prefix!r}; "
                "--chart-single-run asserts that every value column belongs to that exact run. "
                "Export a single-run chart or a run's history CSV."
            )
        name, metric = single_run, column[len(prefix):]
    else:
        if column.count(" - ") > 1:
            raise ValueError(
                f"ambiguous W&B chart label {column!r}: ' - ' may occur in a run or metric name. "
                "For a chart containing only one run, use --chart-single-run with its exact "
                "display name. Otherwise export a run's history CSV. --run cannot resolve "
                "this boundary."
            )
        name, sep, metric = column.partition(" - ")
        if not sep:
            return "", column
    if not metric.strip():
        raise ValueError(f"chart column {column!r} has an empty metric name")
    return name, metric


def read_wandb_chart_csv(
    path: str | Path, run_name: str | None = None, *, step_key: str | None = None,
    chart_single_run: str | None = None,
) -> Run:
    if chart_single_run is not None:
        if run_name is not None:
            raise ValueError("--chart-single-run and --run cannot be combined")
        if not chart_single_run.strip():
            raise ValueError("--chart-single-run requires a nonblank exact run display name")
    p = Path(path)
    run = Run(source_format="wandb", source_path=str(p))
    with p.open(encoding="utf-8-sig", newline="") as fh:
        delimiter = _delimiter(fh, step_key)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=delimiter, strict=True)
        header = next(reader, [])
        rows = list(reader)
    if not header:
        raise ValueError(f"{p}: empty file")
    validate_columns(header)
    if delimiter != ",":
        rows = [[_decimal_point(cell) for cell in row] for row in rows]

    axis = header[0]
    if step_key is not None and step_key != axis:
        raise ValueError(
            f"a W&B chart export has one shared x-axis, {axis!r}; "
            "--step-key must name that column. Export the chart with the desired axis first."
        )
    training_axis = step_key is not None or axis.strip().lower() in _TRAINING_AXES
    if step_key is not None:
        run.meta["step_selection"] = "explicit"
    logging_axis = not training_axis and axis.strip().lower() in _LOGGING_AXES
    integer_axis = training_axis or logging_axis
    by_run: dict[str, dict[str, int]] = {}
    for index, column in enumerate(header[1:], start=1):
        if column.endswith(_BAND_SUFFIXES):
            continue
        name, metric = _split_run(column, chart_single_run)
        run_columns = by_run.setdefault(name, {})
        if metric in run_columns:
            raise ValueError(f"multiple chart columns identify metric {metric!r} in run {name!r}")
        run_columns[metric] = index

    def points(columns: dict[str, int]) -> int:
        cells = (
            row[i] for row in rows if len(row) == len(header)
            and (not integer_axis or try_step(row[0]) is not None)
            for i in columns.values()
        )
        return sum(1 for cell in cells if try_float(cell) is not None)

    counts = {name: points(columns) for name, columns in by_run.items()}
    chosen = select_run(run, counts, run_name, what=f"{p}: W&B chart export")
    columns = by_run[chosen]
    names = metric_names(run, columns)
    run.meta["chart_identity"] = {
        "interpretation": "declared_single_run" if chart_single_run is not None
        else "unambiguous_labels",
        "selected_run": chosen,
        "columns": {metric: header[index] for metric, index in columns.items()},
    }

    run.meta["step_column"] = axis if integer_axis else f"(row index; the x-axis is {axis!r})"
    if logging_axis:
        run.meta["step_axis"] = "logging_step"
    skipped = 0
    for index, row in enumerate(rows):
        if len(row) != len(header):
            skipped += 1
            step: int | None = index
            if integer_axis:
                step = try_step(row[0]) if row else None
            if step is not None:
                unreadable_row(run, names.values(), step)
            continue
        step = index
        wall_time = None
        x = try_float(row[0]) if row else None
        if integer_axis:
            parsed = try_step(row[0])
            if parsed is None:  # NaN or an infinity: not a step
                skipped += 1
                continue
            step = parsed
        elif "time" in axis.lower():
            wall_time = x
        for metric, i in columns.items():
            value = metric_value(run, names[metric], row[i], step=step, wall_time=wall_time)
            if value is not None:
                run.add_point(names[metric], step, value, wall_time=wall_time)
    run.meta["skipped_rows"] = skipped
    return run


def read_wandb_csv(
    path: str | Path, run_name: str | None = None, *, step_key: str | None = None,
    chart_single_run: str | None = None,
) -> Run:
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as fh:
        delimiter = _delimiter(fh, step_key)
        fh.seek(0)
        header = next(csv.reader(fh, delimiter=delimiter), [])
    validate_columns(header)
    if is_chart_export(header):
        return read_wandb_chart_csv(
            p, run_name, step_key=step_key, chart_single_run=chart_single_run,
        )
    if chart_single_run is not None:
        raise ValueError(
            "--chart-single-run applies only to W&B chart CSV exports with band columns"
        )

    run = Run(source_format="wandb", source_path=str(p))
    with p.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter, strict=True)
        fieldnames = reader.fieldnames or []
        step_keys = [key for key in fieldnames if key.strip().lower() == _STEP_COL]
        if len(step_keys) > 1:
            raise ValueError("multiple W&B step columns; keep one _step axis")
        if not step_keys:
            raise ValueError(
                f"{p}: no '_step' column -- is this a W&B history CSV? Expected a run's history "
                "(wandb.Api().run(path).history() written to CSV) or a chart's CSV export."
            )
        if step_key is None:
            training = [
                key for key in fieldnames if key.strip().lower() in _TRAINING_AXES | {"step"}
            ]
            if len(training) > 1:
                raise ValueError(
                    f"multiple W&B training-step columns: {', '.join(training)}; "
                    "choose the counter used by these metrics with --step-key"
                )
            step_key = training[0] if training else step_keys[0]
            if not training:
                run.meta["step_axis"] = "logging_step"
        else:
            _find_step_key(fieldnames, step_key)
            run.meta["step_selection"] = "explicit"
        runtime_key = next((key for key in fieldnames if key.strip() == _RUNTIME_COL), _RUNTIME_COL)
        run.meta["step_column"] = step_key
        if run_name is not None:
            raise ValueError("--run selects a W&B chart export, not a single-run history CSV")
        names = metric_names(run, (
            k for k in fieldnames
            if k != step_key and k is not None and not k.strip().startswith(_SKIP_PREFIXES)
        ))
        skipped = 0
        for raw_row in reader:
            row = dict(raw_row) if delimiter == "," else _decimal_points(dict(raw_row))
            step = try_step(row.get(step_key))
            if step is None:
                skipped += 1
                continue
            if None in row or None in row.values():
                skipped += 1
                unreadable_row(run, names.values(), step)
                continue
            wall_time = try_float(row.get(runtime_key))
            for key, raw_value in row.items():
                # csv.DictReader files the cells beyond the header under the key None.
                if key is None or key == step_key or key.strip().startswith(_SKIP_PREFIXES):
                    continue
                value = metric_value(run, names[key], raw_value, step=step, wall_time=wall_time)
                if value is None:
                    continue
                run.add_point(names[key], step, value, wall_time=wall_time)
    run.meta["skipped_rows"] = skipped
    return run
