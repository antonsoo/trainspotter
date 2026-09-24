"""Weights & Biases run-history CSV reader.

Produced by, from a run page: **Overview -> ... (three-dot menu) -> Download
history CSV**, or programmatically via
`wandb.Api().run(path).history(pandas=False)` written to CSV. Either way you
get columns `_step`, `_runtime`, `_timestamp`, and one column per logged
metric, already namespaced the way you called `wandb.log` (e.g. `train/loss`,
`eval/loss`) -- those namespaced names are already trainspotter's canonical
vocabulary, so no aliasing is needed for them; only the `_`-prefixed W&B
bookkeeping columns get mapped.
"""

from __future__ import annotations

import csv
from pathlib import Path

from trainspotter.model import Run

from .common import normalize_key, try_float

_STEP_COL = "_step"
_RUNTIME_COL = "_runtime"
_SKIP_PREFIXES = ("_",)


def read_wandb_csv(path: str | Path) -> Run:
    p = Path(path)
    run = Run(source_format="wandb", source_path=str(p))
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if _STEP_COL not in fieldnames:
            raise ValueError(
                f"{p}: no '_step' column -- is this a W&B history CSV export? "
                "(Run page -> ... menu -> Download history CSV)"
            )
        run.meta["step_column"] = _STEP_COL
        for i, row in enumerate(reader):
            step_val = try_float(row.get(_STEP_COL))
            step = int(step_val) if step_val is not None else i
            wall_time = try_float(row.get(_RUNTIME_COL))
            for key, raw_value in row.items():
                if key.startswith(_SKIP_PREFIXES):
                    continue
                value = try_float(raw_value)
                if value is None:
                    continue
                run.add_point(normalize_key(key), step, value, wall_time=wall_time)
    return run
