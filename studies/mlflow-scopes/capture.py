"""Native MLflow 3.17.1 logging controls; no training, models or data downloaded.

Values and metric clocks are authored to distinguish run-level aggregates from
model/dataset scopes. The real SDK writes and reads its native SQLite schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
# Configure the default experiment before the SDK creates its native schema.
# No artifact is logged, so this authored URI is never opened or created.
os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = "file:///controlled-mlflow-scopes"

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert mlflow.__version__ == "3.17.1", mlflow.__version__
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output.absolute() / "scopes.sqlite"
    client = MlflowClient(tracking_uri="sqlite:///" + str(database))
    experiment = client.create_experiment(
        "Controlled metric scope evidence", artifact_location="file:///controlled-mlflow-scopes"
    )
    cases = []
    for label, plain_key, qualified, dataset, repeated, two_models in (
        ("unqualified", "loss", False, False, False, False),
        ("mixed-dataset", "loss", True, True, False, False),
        ("coalesced-coordinate", "loss", True, True, True, False),
        ("unqualified-alias", "train_loss", True, True, False, False),
        ("two-models", "loss", True, True, False, True),
        ("model-without-dataset", "loss", True, False, False, False),
    ):
        run = client.create_run(
            experiment, tags={"mlflow.runName": label, "mlflow.user": "controlled-logging-study"}
        ).info.run_id
        models = [
            client.create_logged_model(
                experiment, name=f"{label}-model-{index}", source_run_id=run
            ).model_id
            for index in range(2 if two_models else int(qualified))
        ]
        calls = []
        for index in range(20):
            value = 1 / (1 + index / 10)
            clock = 1_700_000_000_000 + index * 1000
            client.log_metric(run, plain_key, value, timestamp=clock, step=index, synchronous=True)
            calls.append(
                {
                    "key": plain_key,
                    "step": index,
                    "timestamp": clock,
                    "value": value,
                    "model_id": None,
                    "dataset_name": None,
                    "dataset_digest": None,
                }
            )
        if qualified:
            for index in range(20):
                step = index if repeated else index + 20
                value = (
                    1 / (1 + index / 10)
                    if repeated
                    else 10.0 + 10.0 * (index // 10 if two_models else 0)
                )
                model = models[index // 10 if two_models else 0]
                clock = 1_700_000_000_000 + step * 1000
                name, digest = ("held-out", "abcd1234") if dataset else (None, None)
                client.log_metric(
                    run,
                    "loss",
                    value,
                    timestamp=clock,
                    step=step,
                    synchronous=True,
                    model_id=model,
                    dataset_name=name,
                    dataset_digest=digest,
                )
                calls.append(
                    {
                        "key": "loss",
                        "step": step,
                        "timestamp": clock,
                        "value": value,
                        "model_id": model,
                        "dataset_name": name,
                        "dataset_digest": digest,
                    }
                )
        client.set_terminated(run)
        histories = {}
        for key in sorted({call["key"] for call in calls}):
            histories[key] = [
                dict(
                    key=point.key,
                    value=point.value,
                    step=point.step,
                    timestamp=point.timestamp,
                    model_id=point.model_id,
                    dataset_name=point.dataset_name,
                    dataset_digest=point.dataset_digest,
                )
                for point in client.get_metric_history(run, key)
            ]
        cases.append(
            {
                "name": label,
                "run_id": run,
                "model_ids": models,
                "authored_log_calls": calls,
                "sdk_run_histories": histories,
            }
        )
    with sqlite3.connect(database) as db:
        for case in cases:
            case["sql_model_scopes"] = [
                dict(
                    metric=key,
                    model_id=model,
                    dataset_name=name,
                    dataset_digest=digest,
                    recorded_rows=count,
                )
                for key, model, name, digest, count in db.execute(
                    "SELECT metric_name, model_id, dataset_name, dataset_digest, count(*) "
                    "FROM logged_model_metrics WHERE run_id=? GROUP BY metric_name, model_id, "
                    "dataset_name, dataset_digest ORDER BY metric_name, model_id",
                    (case["run_id"],),
                )
            ]
        # The user tag is an authored producer input; no database normalization.
        assert {row[0] for row in db.execute("SELECT user_id FROM runs")} == {
            "controlled-logging-study"
        }
        assert not any(
            "/home/" in str(value) or "/tmp/" in str(value)
            for row in db.execute("SELECT artifact_uri FROM runs")
            for value in row
        )
        assert "/home/" not in "\n".join(db.iterdump())
        assert "/tmp/" not in "\n".join(db.iterdump())
    raw = database.read_bytes()
    result = {
        "sdk": mlflow.__version__,
        "scope": __doc__,
        "normalization": "None; native database bytes retained.",
        "database": {
            "file": database.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "cases": cases,
    }
    (args.output / "oracle.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "runs": len(cases),
                "sdk_run_observations": sum(
                    len(rows) for c in cases for rows in c["sdk_run_histories"].values()
                ),
                "model_rows": sum(
                    scope["recorded_rows"] for c in cases for scope in c["sql_model_scopes"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
