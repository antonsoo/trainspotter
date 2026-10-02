"""Generic CSV / JSONL metric-log reader.

Format: one row/object per logged step, with a step column (`step`,
`global_step`, `iteration` or `iter`) and arbitrary numeric metric columns.
This is the escape hatch for logs trainspotter doesn't have a dedicated
reader for -- dump your framework's log to CSV or JSONL with a step column
and it works.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from trainspotter.model import Run

from .common import build_key_map, try_float, try_step

_STEP_KEYS = ("step", "global_step", "iteration", "iter")


def _find_step_key(keys: Sequence[str]) -> str | None:
    lower = {k.lower(): k for k in keys}
    for candidate in _STEP_KEYS:
        if candidate in lower:
            return lower[candidate]
    return None


def _ingest_row(
    run: Run, row: dict[str, object], step_key: str | None, index: int, names: dict[str, str]
) -> bool:
    """Add one row's metrics to the run, each under its name in `names`. False if the row
    was skipped: its step cell holds something that can't be a step (NaN, an infinity)."""
    step = index
    if step_key is not None and try_float(row.get(step_key)) is not None:
        parsed = try_step(row.get(step_key))
        if parsed is None:
            return False
        step = parsed
    wall_time = None
    wt_raw = row.get("wall_time") if "wall_time" in row else row.get("timestamp")
    if wt_raw is not None:
        wall_time = try_float(wt_raw)
    for key, raw_value in row.items():
        # csv.DictReader files the cells beyond the header under the key None.
        if key is None or (step_key is not None and key == step_key):
            continue
        if key in ("wall_time", "timestamp"):
            continue
        value = try_float(raw_value)
        if value is None:
            continue
        run.add_point(names[key], step, value, wall_time=wall_time)
    return True


def _metric_names(keys: Sequence[str], step_key: str | None) -> dict[str, str]:
    """The name each column is stored under (see `build_key_map`), leaving out the columns
    that are not metrics."""
    not_metrics = {step_key, "wall_time", "timestamp", None}
    return build_key_map(k for k in keys if k not in not_metrics)


def read_csv(path: str | Path) -> Run:
    p = Path(path)
    run = Run(source_format="csv", source_path=str(p))
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        step_key = _find_step_key(fieldnames)
        run.meta["step_column"] = step_key or "(row index)"
        names = _metric_names(fieldnames, step_key)
        skipped = sum(
            not _ingest_row(run, dict(row), step_key, i, names) for i, row in enumerate(reader)
        )
    run.meta["skipped_rows"] = skipped
    return run


def read_jsonl(path: str | Path) -> Run:
    p = Path(path)
    run = Run(source_format="jsonl", source_path=str(p))
    skipped = 0
    rows: list[dict[str, object]] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            rows.append(obj)
    # In order of first appearance, so that which of two rival names wins does not depend on
    # how they sort.
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    step_key = _find_step_key(sorted(all_keys))
    run.meta["step_column"] = step_key or "(row index)"
    names = _metric_names(all_keys, step_key)
    skipped += sum(not _ingest_row(run, row, step_key, i, names) for i, row in enumerate(rows))
    run.meta["skipped_lines"] = skipped
    return run
