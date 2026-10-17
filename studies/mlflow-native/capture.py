"""Write native MLflow stores and retain the native client's full-history oracle.

Training measurements are replayed from the repository's recorded MLP and W&B
examples. MLP replay timestamps are controlled; W&B timestamps are converted to
MLflow's integer milliseconds. No new training or remote server is claimed.

Requires mlflow-skinny, sqlalchemy and alembic in an isolated environment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_MS = 1_700_000_000_000


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    return value


def examples():
    cases = []
    for name in ("baseline", "divergence", "overfitting"):
        path = ROOT / "examples" / f"{name}.trainer_state.json"
        records = json.loads(path.read_text())["log_history"]
        metrics = [
            (key, float(value), BASE_MS + int(record["step"]) * 1000, int(record["step"]))
            for record in records
            for key, value in record.items()
            if key != "step"
        ]
        cases.append(
            (
                name,
                metrics,
                path.relative_to(ROOT).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    for name in ("stable-training", "slow-training"):
        path = ROOT / "examples/wandb-axes" / f"{name}.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        metrics = [
            (
                "train/loss",
                float(row["train/loss"]),
                int((Decimal(row["_timestamp"]) * 1000).to_integral_value()),
                int(row["trainer/global_step"]),
            )
            for row in rows
        ]
        cases.append(
            (
                name,
                metrics,
                path.relative_to(ROOT).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    cases.append(
        (
            "nonfinite",
            [
                ("loss", value, BASE_MS + step * 1000, step)
                for step, value in enumerate(
                    [1.0, 0.8, 0.6, float("nan"), float("inf"), -float("inf"), 0.2]
                )
            ],
            "controlled special-value observations",
            None,
        )
    )
    cases.append(
        (
            "conflicting-steps",
            [("loss", 1 / (step + 1), BASE_MS + step * 1000, step) for step in range(20)]
            + [("loss", 5.0, BASE_MS + 21_000, 7)],
            "controlled conflicting readings",
            None,
        )
    )
    cases.append(
        (
            "default-step",
            [("loss", 1 / (step + 1), BASE_MS + step * 1000, 0) for step in range(20)],
            "controlled observations with no advancing step",
            None,
        )
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for native captures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    from mlflow import __version__
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    stores = []
    for kind in ("sqlite", "files"):
        store = args.out / ("mlflow.db" if kind == "sqlite" else "mlruns")
        uri = "sqlite:///" + str(store.resolve()) if kind == "sqlite" else store.resolve().as_uri()
        client = MlflowClient(tracking_uri=uri)
        experiment = client.create_experiment(
            "Trainspotter native history replay",
            artifact_location="file:///controlled-mlflow-artifacts",
        )
        runs = []
        for name, metrics, source, sha256 in examples():
            run = client.create_run(
                experiment,
                start_time=BASE_MS,
                run_name=name,
                tags={"mlflow.user": "trainspotter-native-study"},
            )
            records = [
                Metric(key, value, timestamp, step) for key, value, timestamp, step in metrics
            ]
            for start in range(0, len(records), 500):
                client.log_batch(run.info.run_id, metrics=records[start : start + 500])
            client.set_terminated(
                run.info.run_id,
                status="FAILED" if name == "divergence" else "FINISHED",
                end_time=BASE_MS + 1_000_000,
            )
            histories = {
                key: sorted(
                    [
                        {"step": item.step, "value": safe(item.value), "timestamp": item.timestamp}
                        for item in client.get_metric_history(run.info.run_id, key)
                    ],
                    key=lambda item: (item["step"], item["timestamp"], str(item["value"])),
                )
                for key in sorted({row[0] for row in metrics})
            }
            runs.append(
                {
                    "name": name,
                    "runId": run.info.run_id,
                    "experimentId": experiment,
                    "source": source,
                    "sourceSha256": sha256,
                    "loggedRows": len(records),
                    "histories": histories,
                }
            )
            print(
                json.dumps({"kind": kind, "run": name, "rows": sum(map(len, histories.values()))}),
                flush=True,
            )
        stores.append({"kind": kind, "path": store.relative_to(args.out).as_posix(), "runs": runs})
    files = [
        {
            "file": path.relative_to(args.out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
    ]
    output = {
        "mlflow": __version__,
        "python": platform.python_version(),
        "population": (
            "Native local SDK replay of retained training measurements, plus explicitly "
            "controlled integrity cases. No server, artifact download or new training."
        ),
        "timestampPolicy": (
            "Controlled one-second spacing for MLP replay; retained W&B wall clocks "
            "rounded to native integer milliseconds."
        ),
        "stores": stores,
        "files": files,
    }
    histories = gzip.compress(json.dumps(safe(output), separators=(",", ":")).encode(), mtime=0)
    (args.out / "histories.json.gz").write_bytes(histories)
    output["nativeHistoryFile"] = {
        "file": "histories.json.gz",
        "sha256": hashlib.sha256(histories).hexdigest(),
    }
    for store in output["stores"]:
        for run in store["runs"]:
            run["histories"] = {
                key: {
                    "observations": len(records),
                    "sha256": hashlib.sha256(
                        json.dumps(records, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                for key, records in run["histories"].items()
            }
    (args.out / "capture.json").write_text(json.dumps(safe(output), indent=2) + "\n")


if __name__ == "__main__":
    main()
