"""Capture actual CPU regression training and fixed-model evaluation passes.

Synthetic data, known evaluation-cohort changes, and explicitly labelled missing
or invalid log records. No Trainspotter imports. Raw predictions and targets let
the replay script check the means independently of PyTorch and the detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def digest_model(model: torch.nn.Module) -> str:
    values = [p.detach().flatten().tolist() for p in model.parameters()]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2718)
    generator = torch.Generator().manual_seed(31415)
    train_x = torch.randn(20, 16, dtype=torch.float64, generator=generator)
    eval_x = torch.randn(128, 16, dtype=torch.float64, generator=generator)
    teacher = torch.linspace(-0.3, 0.3, 16, dtype=torch.float64)
    train_y = train_x @ teacher + torch.randn(20, dtype=torch.float64, generator=generator)
    eval_y = eval_x @ teacher + torch.randn(128, dtype=torch.float64, generator=generator)
    model = torch.nn.Linear(16, 1, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    history = []
    scores = []
    snapshots = {}
    for step in range(301):
        if step:
            optimizer.zero_grad()
            loss = ((model(train_x).flatten() - train_y) ** 2).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_predictions = model(train_x).flatten()
            eval_predictions = model(eval_x).flatten()
            history.append(
                {
                    "step": step,
                    "loss": ((train_predictions - train_y) ** 2).mean().item(),
                    "eval_loss": ((eval_predictions - eval_y) ** 2).mean().item(),
                }
            )
            if step in (*range(6), 60, 300):
                snapshots[step] = digest_model(model)
                for cohort, predictions, targets in (
                    ("train", train_predictions, train_y),
                    ("eval", eval_predictions, eval_y),
                ):
                    scores.extend(
                        {"step": step, "cohort": cohort, "example": i, "prediction": p, "target": y}
                        for i, (p, y) in enumerate(
                            zip(predictions.tolist(), targets.tolist(), strict=True)
                        )
                    )
    fixed_hash = digest_model(model)
    with torch.no_grad():
        per_example = (model(eval_x).flatten() - eval_y) ** 2
    # Deliberate cohort selection, NOT random evaluation noise. Same frozen
    # model, different evaluation data: aggregates alone cannot tell the cause.
    hard_indices = torch.argsort(per_example, descending=True)[:32]
    fixed = []
    switched = []
    for evaluation_pass in range(20):
        with torch.no_grad():
            full = ((model(eval_x).flatten() - eval_y) ** 2).mean().item()
            hard = (
                ((model(eval_x[hard_indices]).flatten() - eval_y[hard_indices]) ** 2).mean().item()
            )
        assert digest_model(model) == fixed_hash
        fixed.append({"step": evaluation_pass, "eval_loss": full})
        switched.append(
            {"step": evaluation_pass, "eval_loss": hard if evaluation_pass % 2 else full}
        )
    artifacts = {
        "regression-full.csv": (history, "Actual training, optimizer updates 0..300"),
        "smooth-six.csv": (history[:6], "First six observations of regression-full.csv"),
        "late-training.csv": (
            [
                {
                    "step": r["step"],
                    "loss": r["loss"] if r["step"] >= 100 else "",
                    "eval_loss": r["eval_loss"] if r["step"] <= 60 else "",
                }
                for r in history
            ],
            "Derived: evaluation ends at 60; all training readings before 100 removed",
        ),
        "fixed-repeat.csv": (fixed, "Same frozen model and same full evaluation cohort, 20 passes"),
        "fixed-cohort-switch.csv": (
            switched,
            "Same frozen model; alternate full cohort and highest-loss 32 examples",
        ),
        "nan-gap.csv": (
            [{**r, "eval_loss": "NaN" if r["step"] == 100 else r["eval_loss"]} for r in history],
            "Derived: evaluation at step 100 replaced by NaN",
        ),
        "nan-final.csv": (
            [{**r, "eval_loss": "NaN" if r["step"] == 300 else r["eval_loss"]} for r in history],
            "Derived: final evaluation replaced by NaN",
        ),
        "predictions.csv": (scores, "Actual predictions and targets at eight checkpoints"),
    }
    files = {}
    for name, (rows, description) in artifacts.items():
        path = args.out / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        files[name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "description": description,
        }
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "data_seed": 31415,
        "model_seed": 2718,
        "data": ("Synthetic linear regression, 16 features, "
                 "Gaussian label noise sd=1; 20 train, 128 eval"),
        "model": "torch.nn.Linear(16, 1, dtype=torch.float64)",
        "optimizer": "SGD(lr=0.03)",
        "updates": 300,
        "step_units": {
            "regression": "completed optimizer updates; loss evaluated after update",
            "fixed-*": "evaluation pass ordinal; optimizer remains at step 300",
        },
        "snapshot_sha256": snapshots,
        "fixed_snapshot_sha256": fixed_hash,
        "hard_example_indices": hard_indices.tolist(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": files,
    }
    (args.out / "capture.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "first_six": history[:6],
                "best": min(history, key=lambda r: r["eval_loss"]),
                "eval_end": history[60],
                "final": history[-1],
                "fixed_switch": switched[:2],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
