"""Capture learning rates actually used by small CPU PyTorch training runs.

The data and deliberately misconfigured schedules are controlled examples,
not production failures. No Trainspotter code is imported here. See README.md
for the pinned capture environment and the independent CLI replay command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

STEPS = 240
RESUME_STEP = 200
WARMUP = 40
PEAK_LR = 0.1


def linear_factor(step: int) -> float:
    return min(step / WARMUP, max(0.0, (STEPS - step) / (STEPS - WARMUP)))


def make_training(scenario: str) -> tuple[Any, Any, Any]:
    torch.manual_seed(2718)
    model = torch.nn.Linear(8, 1, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=PEAK_LR, momentum=0.9)
    if scenario.startswith("linear"):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_factor)
    elif scenario == "cosine-control":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    elif scenario == "cosine-overrun":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS // 2)
    elif scenario == "cosine-restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=80)
    elif scenario == "one-cycle-control":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=PEAK_LR, total_steps=STEPS, pct_start=0.25, cycle_momentum=False
        )
    elif scenario == "step-decay-control":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    else:
        raise ValueError(scenario)
    return model, optimizer, scheduler


def capture(scenario: str) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    generator = torch.Generator().manual_seed(31415)
    inputs = torch.randn(128, 8, dtype=torch.float64, generator=generator)
    teacher = torch.tensor([1, -2, 1, 0.5, 0, -0.5, 2, -1], dtype=torch.float64)
    targets = (inputs @ teacher > 0).to(torch.float64).unsqueeze(1)
    model, optimizer, scheduler = make_training(scenario)
    metadata: dict[str, Any] = {
        "scheduler": type(scheduler).__name__,
        "initial_scheduler_state": scheduler.state_dict(),
        "resume": None,
    }
    rows = []
    for step in range(STEPS):
        if step == RESUME_STEP and scenario in ("linear-resume-restored", "linear-resume-reset"):
            buffer = io.BytesIO()
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                buffer,
            )
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True)
            model, optimizer, scheduler = make_training(scenario)
            model.load_state_dict(checkpoint["model"])
            if scenario == "linear-resume-restored":
                scheduler.load_state_dict(checkpoint["scheduler"])
            # Construct the scheduler before restoring the optimizer: constructing
            # LambdaLR itself changes the optimizer's learning rate.
            optimizer.load_state_dict(checkpoint["optimizer"])
            metadata["resume"] = {
                "completed_updates": step,
                "checkpoint_scheduler_epoch": checkpoint["scheduler"]["last_epoch"],
                "resumed_scheduler_epoch": scheduler.last_epoch,
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "scheduler_state_loaded": scenario == "linear-resume-restored",
            }
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
        loss.backward()
        # This is the rate applied by the update immediately below, not the rate
        # scheduled for the next update or an independently generated curve.
        rows.append(
            {
                "step": step,
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "scheduler_epoch": scheduler.last_epoch,
            }
        )
        optimizer.step()
        scheduler.step()
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observations = {}
    scenarios = {}
    for name in (
        "linear-control",
        "linear-resume-restored",
        "linear-resume-reset",
        "cosine-control",
        "cosine-overrun",
        "cosine-restarts",
        "one-cycle-control",
        "step-decay-control",
    ):
        rows, metadata = capture(name)
        observations[name] = rows
        scenarios[name] = metadata
    assert observations["linear-control"] == observations["linear-resume-restored"]
    files = {}
    for name, rows in observations.items():
        path = args.out / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[path.name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # Sparse logging is an explicit derivation of a captured run, not another run.
    for origin in ("linear-control", "linear-resume-reset", "cosine-overrun"):
        path = args.out / f"{origin}-sparse.csv"
        rows = [row for row in observations[origin] if int(row["step"]) % 10 == 0]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[path.name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "derived_from": origin,
            "every_nth_update": 10,
        }
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": 1,
        "data": "128 fixed synthetic binary classification examples, 8 features",
        "model": "torch.nn.Linear(8, 1, dtype=torch.float64)",
        "optimizer": "SGD(lr=0.1, momentum=0.9)",
        "steps": STEPS,
        "warmup": WARMUP,
        "resume_step": RESUME_STEP,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": scenarios,
        "files": files,
    }
    (args.out / "capture.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Captured {len(scenarios)} training runs and {len(files)} CSVs in {args.out}")


if __name__ == "__main__":
    main()
