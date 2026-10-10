"""Record training and logging counters with the real W&B SDK, entirely offline.

Install the capture-only dependencies in a separate environment:
    uv pip install --python /path/to/venv wandb==0.22.3 numpy==2.5.3
    /path/to/venv/bin/python examples/capture_wandb_axes.py --output /tmp/wandb-axes

The pinned SDK includes the decoder for its local .wandb files. Retain every
history protobuf as JSON, all measured updates, and CSVs derived from those
history records. No server-side history export or network model is involved.
The data and pacing are deliberately controlled; this is not a GPU benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from google.protobuf.json_format import MessageToDict
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture(work: Path, name: str, *, late_delay: bool) -> dict[str, Any]:
    rng = np.random.default_rng(20261009)
    x = rng.normal(size=(128, 8))
    y = (x @ rng.normal(size=8) > 0).astype(float)
    weights = np.zeros(8)

    def loss() -> float:
        logits = x @ weights
        return float(np.mean(np.logaddexp(0, logits) - y * logits))

    run = wandb.init(
        mode="offline", project="trainspotter-axis-capture", name=name, dir=str(work),
        settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True,
                                console="off", quiet=True),
    )
    run.define_metric("trainer/global_step")
    run.define_metric("train/*", step_metric="trainer/global_step")
    run_dir = Path(run.dir).parent
    observations = []
    try:
        started = time.perf_counter()
        for step in range(1201):
            before = time.perf_counter()
            pace = 0.012 if late_delay and step > 200 else 0.004
            if step:
                probability = 1 / (1 + np.exp(-(x @ weights)))
                weights -= 0.02 * (x.T @ (probability - y) / len(y))
                time.sleep(max(0, pace - (time.perf_counter() - before)))
            elapsed = time.perf_counter() - started
            value = loss()
            logged = step % (4 if step <= 200 else 20) == 0
            observations.append({"step": step, "elapsed": elapsed, "loss": value,
                                 "logged": logged, "requested_seconds": pace if step else None})
            if logged:
                run.log({"trainer/global_step": step, "train/loss": value,
                         "capture/elapsed": elapsed})
    finally:
        run.finish()
    native, = run_dir.glob("*.wandb")
    store = DataStore()
    store.open_for_scan(str(native))
    records = []
    rows = []
    try:
        while (data := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(data)
            if record.HasField("history"):
                records.append(MessageToDict(record.history, preserving_proto_field_name=True))
                row = {}
                for item in record.history.item:
                    key = item.key or "/".join(item.nested_key)
                    assert key not in row, f"duplicate history key {key}"
                    row[key] = json.loads(item.value_json)
                rows.append(row)
    finally:
        store.close()
    return {"observations": observations, "records": records, "rows": rows,
            "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, help="Temporary local SDK files (never synced)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wandb-axis-", dir=args.work) as scratch:
        captured = {
            name: capture(Path(scratch), name, late_delay=name == "slow")
            for name in ("stable", "slow")
        }
    files = {}
    for name, source in captured.items():
        history_path = args.output / f"{name}-history.jsonl"
        history_path.write_text("".join(json.dumps(row) + "\n" for row in source["records"]))
        source.pop("records")
        rows = source.pop("rows")
        variants = {
            "training": rows,
            "logging": [{k: v for k, v in row.items() if k != "trainer/global_step"}
                        for row in rows],
            "custom": [{("optimizer/updates" if k == "trainer/global_step" else k): v
                        for k, v in row.items()} for row in rows],
        }
        for suffix, values in variants.items():
            write_csv(args.output / f"{name}-{suffix}.csv", values)
    for path in sorted(args.output.iterdir()):
        if path.suffix in (".csv", ".jsonl"):
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Paced logistic regression on synthetic data, logged with W&B offline",
        "python": platform.python_version(), "numpy": np.__version__, "wandb": wandb.__version__,
        "system": platform.system(), "machine": platform.machine(),
        "logging": "Every 4 updates through step 200; every 20 updates thereafter, to step 1200",
        "pacing": "0.004 s/update; slow case requests 0.012 s/update after step 200",
        "history": "Complete history protobufs decoded with the pinned SDK, not cloud CSV exports",
        "files": files, "captures": captured,
    }
    (args.output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": files,
                      "wandb": wandb.__version__}, indent=2))


if __name__ == "__main__":
    main()
