"""Hugging Face `transformers.Trainer` reader.

`Trainer` writes `trainer_state.json` on every checkpoint. The field that
matters is `log_history`: a flat list of dicts, one per logged event.
Training-step entries look like
`{"loss": 2.31, "learning_rate": 5e-5, "epoch": 0.1, "step": 10, "grad_norm": 1.2}`;
evaluation entries look like
`{"eval_loss": 2.1, "eval_accuracy": 0.4, "epoch": 0.1, "step": 10}`.
There is no reliable per-step wall-clock timestamp in this format (only
aggregate `train_runtime`/`train_samples_per_second` at the very end), so
`step_time`/`throughput` are usually absent for HF logs -- the throughput
detector degrades gracefully when they are.

`Trainer.train()` appends exactly one more entry after the last real log:
a run-level summary with keys like `train_runtime` and `train_loss` (the
*average* loss over the whole run, not a per-step reading), reusing the
final `step` number. Verified against a real `transformers.Trainer` run
(see `examples/generate_transformers_example.py`): the first version of
this reader merged that average straight into the `train/loss` series --
since `train_loss` is also `loss`'s alias -- which showed up as a fake
last-step "spike" back down or up to the run average. `train_runtime`
only ever appears on that one summary entry, so it's used as the marker
to route the whole entry to run metadata instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from trainspotter.model import Run

from .common import normalize_key, try_float

_NON_METRIC_KEYS = {"step", "total_flos"}
_SUMMARY_MARKER_KEY = "train_runtime"


def read_hf_trainer_state(path: str | Path) -> Run:
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and "log_history" in data:
        log_history = data["log_history"]
        meta: dict[str, object] = {
            k: v for k, v in data.items() if k != "log_history" and not isinstance(v, dict | list)
        }
    elif isinstance(data, list):
        log_history = data
        meta = {}
    else:
        raise ValueError(
            f"{p}: expected an object with a 'log_history' list (trainer_state.json) "
            "or a bare list of log entries"
        )

    run = Run(source_format="hf", source_path=str(p), meta=meta)
    skipped = 0
    for entry in log_history:
        if not isinstance(entry, dict) or "step" not in entry:
            skipped += 1
            continue
        if _SUMMARY_MARKER_KEY in entry:
            # The run-level training summary, not a per-step log point.
            run.meta["train_summary"] = {k: v for k, v in entry.items() if k != "step"}
            continue
        step_val = try_float(entry.get("step"))
        if step_val is None:
            skipped += 1
            continue
        step = int(step_val)
        for key, raw_value in entry.items():
            if key in _NON_METRIC_KEYS:
                continue
            value = try_float(raw_value)
            if value is None:
                continue
            canon = normalize_key(key)
            run.add_point(canon, step, value)
    run.meta["skipped_entries"] = skipped
    return run
