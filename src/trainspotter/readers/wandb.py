"""Weights & Biases CSV readers: a run's history, and a chart's export.

**Run history.** From the public API, `wandb.Api().run(path).history()`
written to CSV: columns `_step`, `_runtime`, `_timestamp`, and one column per
logged metric, named the way you called `wandb.log` (e.g. `train/loss`,
`eval/loss`).

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

from .common import metric_names, metric_value, select_run, try_float, try_step, validate_columns
from .tabular import _decimal_point, _decimal_points, _delimiter

_STEP_COL = "_step"
_RUNTIME_COL = "_runtime"
_SKIP_PREFIXES = ("_",)
_BAND_SUFFIXES = ("__MIN", "__MAX")
# What the x-axis column is called when it is a step count, lower-cased.
_STEP_AXES = {"step", "_step", "global_step", "trainer/global_step", "train/global_step"}
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


def _split_run(column: str) -> tuple[str, str]:
    """`<run name> - <metric>`, cut at the last separator: a run's display name may hold one."""
    run, sep, metric = column.rpartition(" - ")
    return (run, metric) if sep else ("", column)


def read_wandb_chart_csv(path: str | Path, run_name: str | None = None) -> Run:
    p = Path(path)
    run = Run(source_format="wandb", source_path=str(p))
    with p.open(encoding="utf-8-sig", newline="") as fh:
        delimiter = _delimiter(fh.readline())
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
    by_run: dict[str, dict[str, int]] = {}
    for index, column in enumerate(header[1:], start=1):
        if column.endswith(_BAND_SUFFIXES):
            continue
        name, metric = _split_run(column)
        by_run.setdefault(name, {})[metric] = index

    def points(columns: dict[str, int]) -> int:
        cells = (
            row[i] for row in rows if len(row) == len(header)
            and (axis.strip().lower() not in _STEP_AXES or try_step(row[0]) is not None)
            for i in columns.values()
        )
        return sum(1 for cell in cells if try_float(cell) is not None)

    counts = {name: points(columns) for name, columns in by_run.items()}
    chosen = select_run(run, counts, run_name, what=f"{p}: W&B chart export")
    columns = by_run[chosen]
    names = metric_names(run, columns)

    steps_are_steps = axis.strip().lower() in _STEP_AXES
    run.meta["step_column"] = axis if steps_are_steps else f"(row index; the x-axis is {axis!r})"
    skipped = 0
    for index, row in enumerate(rows):
        if len(row) != len(header):
            skipped += 1
            continue
        step = index
        wall_time = None
        x = try_float(row[0]) if row else None
        if steps_are_steps:
            parsed = try_step(row[0])
            if parsed is None:  # NaN or an infinity: not a step
                skipped += 1
                continue
            step = parsed
        elif "time" in axis.lower():
            wall_time = x
        for metric, i in columns.items():
            value = metric_value(run, names[metric], row[i])
            if value is not None:
                run.add_point(names[metric], step, value, wall_time=wall_time)
    run.meta["skipped_rows"] = skipped
    return run


def read_wandb_csv(path: str | Path, run_name: str | None = None) -> Run:
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as fh:
        delimiter = _delimiter(fh.readline())
        fh.seek(0)
        header = next(csv.reader(fh, delimiter=delimiter), [])
    validate_columns(header)
    if is_chart_export(header):
        return read_wandb_chart_csv(p, run_name)

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
        step_key = step_keys[0]
        runtime_key = next((key for key in fieldnames if key.strip() == _RUNTIME_COL), _RUNTIME_COL)
        run.meta["step_column"] = step_key
        if run_name is not None:
            raise ValueError("--run selects a W&B chart export, not a single-run history CSV")
        names = metric_names(run, (
            k for k in fieldnames if k is not None and not k.strip().startswith(_SKIP_PREFIXES)
        ))
        skipped = 0
        for raw_row in reader:
            row = dict(raw_row) if delimiter == "," else _decimal_points(dict(raw_row))
            step = try_step(row.get(step_key))
            if step is None or None in row or None in row.values():
                skipped += 1
                continue
            wall_time = try_float(row.get(runtime_key))
            for key, raw_value in row.items():
                # csv.DictReader files the cells beyond the header under the key None.
                if key is None or key.strip().startswith(_SKIP_PREFIXES):
                    continue
                value = metric_value(run, names[key], raw_value)
                if value is None:
                    continue
                run.add_point(names[key], step, value, wall_time=wall_time)
    run.meta["skipped_rows"] = skipped
    return run
