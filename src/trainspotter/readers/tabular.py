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
import re
from collections.abc import Sequence
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import TextIO

from trainspotter.model import Run

from .common import (
    metric_names,
    metric_value,
    try_float,
    try_step,
    unique_object,
    unreadable_row,
    validate_columns,
)

_STEP_KEYS = ("step", "global_step", "iteration", "iter")


def _find_step_key(keys: Sequence[str], step_key: str | None = None) -> str | None:
    if step_key is not None:
        if step_key not in keys:
            raise ValueError(
                f"step column {step_key!r} not found; --step-key must name an exact column"
            )
        return step_key
    lower = {k.strip().lower(): k for k in keys}
    candidates = [k for k in keys if k.strip().lower() in _STEP_KEYS]
    if len(candidates) > 1:
        raise ValueError(f"multiple step columns: {', '.join(candidates)}; keep one step axis")
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
    if step_key is not None:
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
        value = metric_value(run, names[key], raw_value, step=step, wall_time=wall_time)
        if value is None:
            continue
        run.add_point(names[key], step, value, wall_time=wall_time)
    return True


def _metric_names(run: Run, keys: Sequence[str], step_key: str | None) -> dict[str, str]:
    """The name each column is stored under (see `build_key_map`), leaving out the columns
    that are not metrics."""
    not_metrics = {step_key, "wall_time", "timestamp", None}
    return metric_names(run, (k for k in keys if k not in not_metrics))


def _delimiter(source: TextIO, step_key: str | None = None) -> str:
    """Infer comma/semicolon/tab from complete records, then rewind the source.

    A run or metric name can contain the other separators, even unquoted: a
    comma writer need not quote semicolons. Header width alone therefore cannot
    establish the delimiter. Prefer the candidate whose first 32 data records
    agree with its header. A declared/recognized coordinate column constrains
    the header first, so damaged rows cannot replace a known step axis with a
    punctuation fragment. Keep malformed-row handling in the actual reader.
    """
    candidates: list[tuple[str, list[str], int, int]] = []
    try:
        for delimiter in (",", ";", "\t"):
            source.seek(0)
            reader = csv.reader(source, delimiter=delimiter, strict=True)
            try:
                header = next(reader, [])
            except csv.Error:
                continue
            matches = mismatches = 0
            try:
                for row in islice(reader, 32):
                    if not row:
                        continue
                    if len(row) == len(header):
                        matches += 1
                    else:
                        mismatches += 1
            except csv.Error:
                # A bad row remains an error when the selected reader replays
                # it. Do not turn a probe failure into an accepted partial file.
                mismatches += 1
            candidates.append((delimiter, header, matches, mismatches))
        multiple = [candidate for candidate in candidates if len(candidate[1]) > 1]
        if not multiple:
            return ","
        axes = {step_key} if step_key is not None else {*_STEP_KEYS, "_step"}
        with_axis = [
            candidate for candidate in multiple
            if candidate[2] > 0
            and any((key if step_key is not None else key.strip().lower()) in axes
                    for key in candidate[1])
        ]
        if with_axis:
            multiple = with_axis
        best_fit = max((matches, -mismatches) for _, _, matches, mismatches in multiple)
        best = [c for c in multiple if (c[2], -c[3]) == best_fit]
        if len(best) == 1:
            return best[0][0]
        # An empty or entirely damaged sample cannot improve on header-only
        # evidence. Preserve the existing width preference and row diagnostics.
        if best_fit[0] == 0:
            return max(best, key=lambda candidate: len(candidate[1]))[0]
        raise ValueError(
            "ambiguous CSV delimiter: multiple separators fit the sampled records; "
            "export a fully quoted CSV so punctuation in labels stays unambiguous"
        )
    finally:
        source.seek(0)


_DECIMAL_COMMA = re.compile(r"[+-]?\d+,\d+(?:[eE][+-]?\d+)?")


def _decimal_point(value: str) -> str:
    return value.strip().replace(",", ".") if _DECIMAL_COMMA.fullmatch(value.strip()) else value


def _decimal_points(row: dict[str, object]) -> dict[str, object]:
    """`0,75` as `0.75`, for a file whose delimiter is not the comma."""
    return {
        key: _decimal_point(value) if isinstance(value, str) else value
        for key, value in row.items()
    }


def read_csv(path: str | Path, *, step_key: str | None = None) -> Run:
    p = Path(path)
    run = Run(source_format="csv", source_path=str(p))
    if step_key is not None:
        run.meta["step_selection"] = "explicit"
    # utf-8-sig: Excel's "CSV UTF-8" starts with a byte-order mark, which would otherwise be
    # part of the first column's name (so "step" was no longer the step column).
    with p.open(encoding="utf-8-sig", newline="") as fh:
        delimiter = _delimiter(fh, step_key)
        reader = csv.DictReader(fh, delimiter=delimiter, strict=True)
        fieldnames = reader.fieldnames or []
        validate_columns(fieldnames)
        step_key = _find_step_key(fieldnames, step_key)
        run.meta["step_column"] = step_key or "(row index)"
        run.meta["columns"] = fieldnames
        names = _metric_names(run, fieldnames, step_key)
        rows = (dict(row) if delimiter == "," else _decimal_points(dict(row)) for row in reader)
        skipped = 0
        for index, row in enumerate(rows):
            if None in row or None in row.values():
                skipped += 1
                step = index if step_key is None else try_step(row.get(step_key))
                if step is not None:
                    unreadable_row(run, names.values(), step)
            elif not _ingest_row(run, row, step_key, index, names):
                skipped += 1
    run.meta["skipped_rows"] = skipped
    return run


def read_jsonl(path: str | Path, *, step_key: str | None = None) -> Run:
    p = Path(path)
    run = Run(source_format="jsonl", source_path=str(p))
    if step_key is not None:
        run.meta["step_selection"] = "explicit"
    skipped = 0
    # Keep rejected nonblank records in place. Filtering before enumeration
    # would compact an ordinal axis and connect trajectories across lost data.
    rows: list[dict[str, object] | None] = []
    with p.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line, object_pairs_hook=unique_object, parse_float=Decimal)
            except (ValueError, RecursionError):
                skipped += 1
                rows.append(None)
                continue
            if not isinstance(obj, dict):
                skipped += 1
                rows.append(None)
                continue
            rows.append(obj)
    # In order of first appearance, so that which of two rival names wins does not depend on
    # how they sort.
    all_keys = list(dict.fromkeys(k for row in rows if row is not None for k in row))
    step_key = _find_step_key(all_keys, step_key)
    run.meta["step_column"] = step_key or "(row index)"
    names = _metric_names(run, all_keys, step_key)
    for index, row in enumerate(rows):
        if row is None:
            if step_key is None:
                unreadable_row(run, names.values(), index)
        elif not _ingest_row(run, row, step_key, index, names):
            skipped += 1
    run.meta["skipped_lines"] = skipped
    return run
