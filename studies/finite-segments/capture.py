"""Controlled CPU training with a logging outage and checkpoint restoration.

Capture only: torch==2.8.0+cpu, Python 3.12. No model/data downloads.
These are controlled experiments, not production training evidence.
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
    optimizer = torch.optim.SGD([weight], lr=0.05)
    checkpoint = None
    rows = []
    for step in range(180):
        if step == 59:
            checkpoint = weight.detach().clone()
        if step == 70:
            with torch.no_grad():
                weight.fill_(3.0)
        if step == 95:
            assert checkpoint is not None
            with torch.no_grad():
                weight.copy_(checkpoint)
        optimizer.zero_grad()
        loss = 1 + weight.square() / 2
        actual = float(loss.detach())
        # A deliberately broken logging reduction on an empty tensor. The
        # training objective and optimizer keep running and are recorded too.
        displayed = torch.empty(0, dtype=torch.float64).mean() if (
            60 <= step < 70 or 90 <= step < 100
        ) else loss.detach()
        rows.append({"step": step, "loss": float(displayed),
                     "objective": actual, "weight": float(weight.detach())})
        loss.backward()
        optimizer.step()
    for filename, fields in (("observed.csv", ["step", "loss"]),
                             ("objective.csv", ["step", "objective", "weight"])):
        with (args.out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "kind": "controlled CPU PyTorch experiment with deliberate logging fault",
        "python": platform.python_version(), "torch": torch.__version__,
        "objective": "1 + w**2/2; float64 SGD; w starts at 3; lr=0.05; 180 observations",
        "events": [
            {"step": 59, "event": "save weight checkpoint before optimizer update"},
            {"step": 70, "event": "replace weight with 3 before observing loss"},
            {"step": 95, "event": "restore saved weight before observing loss"},
        ],
        "logging_fault": "empty tensor mean replaces logged loss at steps 60-69 and 90-99",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(args.out.glob("*.csv"))},
    }
    (args.out / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
