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

from .common import normalize_key, try_float

_STEP_KEYS = ("step", "global_step", "iteration", "iter")


def _find_step_key(keys: Sequence[str]) -> str | None:
    lower = {k.lower(): k for k in keys}
    for candidate in _STEP_KEYS:
        if candidate in lower:
            return lower[candidate]
    return None


def _ingest_row(run: Run, row: dict[str, object], step_key: str | None, index: int) -> None:
    if step_key is not None:
        step_val = try_float(row.get(step_key))
        step = int(step_val) if step_val is not None else index
    else:
        step = index
    wall_time = None
    wt_raw = row.get("wall_time") if "wall_time" in row else row.get("timestamp")
    if wt_raw is not None:
        wall_time = try_float(wt_raw)
    for key, raw_value in row.items():
        if step_key is not None and key == step_key:
            continue
        if key in ("wall_time", "timestamp"):
            continue
        value = try_float(raw_value)
        if value is None:
            continue
        run.add_point(normalize_key(key), step, value, wall_time=wall_time)


def read_csv(path: str | Path) -> Run:
    p = Path(path)
    run = Run(source_format="csv", source_path=str(p))
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        step_key = _find_step_key(fieldnames)
        run.meta["step_column"] = step_key or "(row index)"
        for i, row in enumerate(reader):
            _ingest_row(run, dict(row), step_key, i)
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
    all_keys = {k for row in rows for k in row}
    step_key = _find_step_key(sorted(all_keys))
    run.meta["step_column"] = step_key or "(row index)"
    run.meta["skipped_lines"] = skipped
    for i, row in enumerate(rows):
        _ingest_row(run, row, step_key, i)
    return run
