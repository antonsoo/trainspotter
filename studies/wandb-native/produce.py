"""Capture actual NumPy optimizer runs through the native offline W&B SDK.

Synthetic regression data, an authored learning-rate fault and a deliberate
logging gap. No account, network sync, pretrained model or quality benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_ERROR_REPORTING"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import numpy as np  # noqa: E402
import wandb  # noqa: E402


def safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def emit(run: wandb.Run, calls: list[dict[str, object]], payload: dict[str, object]) -> None:
    calls.append({key: safe(value) for key, value in payload.items() if key != "logger.note"})
    run.log(payload)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    generator = np.random.default_rng(29114)
    features = generator.normal(size=(128, 8))
    evaluation = generator.normal(size=(64, 8))
    teacher = generator.normal(size=8)
    targets = features @ teacher + generator.normal(scale=0.05, size=128)
    evaluation_targets = evaluation @ teacher + generator.normal(scale=0.05, size=64)
    cases = []
    for case in ("stable", "unstable-lr", "logged-null", "custom-counter"):
        directory = out / case
        directory.mkdir()
        settings = wandb.Settings(
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
            console="off",
            host="synthetic-cpu",
            username="native-study",
        )
        run = wandb.init(
            mode="offline",
            dir=directory,
            project="trainspotter-offline-study",
            id="native-" + case,
            name="NumPy regression: " + case,
            settings=settings,
        )
        axis = "optimizer_step" if case == "custom-counter" else "train/global_step"
        run.define_metric(axis)
        run.define_metric("train/*", step_metric=axis)
        run.define_metric("eval/*", step_metric=axis)
        weights = np.zeros(8)
        calls = []
        true_losses = []

        with np.errstate(over="ignore", invalid="ignore"):
            for step in range(1, 401):
                lr = 4.0 if case == "unstable-lr" and step >= 160 else 0.05
                residual = features @ weights - targets
                gradient = 2 * features.T @ residual / len(features)
                weights -= lr * gradient
                loss = float(np.mean((features @ weights - targets) ** 2))
                true_losses.append(safe(loss))
                if step % 5 == 0:
                    logged = None if case == "logged-null" and step == 240 else loss
                    payload: dict[str, object] = {
                        axis: step,
                        "train/loss": logged,
                        "train/learning_rate": lr,
                        "train/grad_norm": float(np.linalg.norm(gradient)),
                        "epoch": step / 20,
                    }
                    if step == 100:
                        # A real SDK-written history record spanning several
                        # 32 KiB blocks; the large text is metadata, not a metric.
                        payload["logger.note"] = "fragment-control-" + "x" * 100_000
                    emit(run, calls, payload)
                if step % 20 == 0:
                    emit(
                        run,
                        calls,
                        {
                            axis: step,
                            "eval/loss": float(
                                np.mean((evaluation @ weights - evaluation_targets) ** 2)
                            ),
                        },
                    )
        source = Path(run.settings.sync_file)
        run.finish()
        destination = directory / "run.wandb"
        shutil.copyfile(source, destination)
        data = destination.read_bytes()
        # Inspect before retaining these controlled native captures publicly.
        # The SDK metadata flags above leave machine paths out of the stream.
        assert not any(token in data for token in (b"/home/", b"/tmp/", b"/mnt/", b"antonsoloviev"))
        reference = {
            "case": case,
            "step_key": axis,
            "optimizer_updates": 400,
            "model": "eight-weight linear regression, full-batch MSE gradient descent",
            "learning_rate_fault": case == "unstable-lr",
            "deliberately_missing_logged_loss_step": 240 if case == "logged-null" else None,
            "actual_training_losses": true_losses,
            "log_calls": calls,
            "fragment_control": {
                "training_step": 100,
                "field": "logger.note",
                "characters": 100_017,
            },
        }
        (directory / "reference.json").write_text(
            json.dumps(reference, indent=2, allow_nan=False) + "\n"
        )
        cases.append(
            {
                "case": case,
                "file": f"{case}/run.wandb",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    (directory / "reference.json").read_bytes()
                ).hexdigest(),
                "sdk_run_id": "native-" + case,
                "logging_calls": len(calls),
            }
        )
    receipt = {
        "scope": "synthetic training, native offline logs and explicit optimizer/logger controls",
        "seed": 29114,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "wandb", "protobuf")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
