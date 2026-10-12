"""Controlled scalar SGD with two intended MultiStepLR drops; no external data.

Capture only: PyTorch 2.8.0+cpu. Faulty exports change one logged LR cell,
never the optimizer configuration or the retained objective/weight records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("choose an empty capture directory")
    torch.set_num_threads(1)
    weight = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
    records = []
    for step in range(120):
        optimizer.zero_grad()
        loss = 1 + weight.square() / 2
        records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight": float(weight.detach()),
            }
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
    variants = {
        "complete": None,
        "unreadable": "LOGGER_ERROR",
        "nan": "nan",
        "positive-inf": "inf",
        "negative-inf": "-inf",
        "negative": "-1",
    }
    for name, bad in variants.items():
        with (args.out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["step", "loss", "learning_rate"],
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                row = dict(record)
                if bad is not None and row["step"] == 10:
                    row["learning_rate"] = bad
                writer.writerow(row)
    with (args.out / "optimizer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["step", "loss", "learning_rate", "weight"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "kind": "controlled CPU optimizer capture with deliberately damaged LR export cells",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "objective": "1 + w**2/2, float64 SGD, initial w=3, initial lr=0.1",
        "scheduler": "MultiStepLR, milestones [40,80], gamma=0.5; called after optimizer updates",
        "observation": "120 observations of loss, current LR and weight before optimizer updates",
        "fault": "Replace only the step-10 learning_rate cell; optimizer and loss unchanged",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.glob("*.csv"))
        },
    }
    (args.out / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {len(records)} optimizer observations and {len(variants)} log exports.")


if __name__ == "__main__":
    main()
