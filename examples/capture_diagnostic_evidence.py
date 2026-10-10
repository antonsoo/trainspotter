"""Record small CPU PyTorch experiments that separate observations from causes.

Capture-only dependency: torch==2.8.0+cpu, from PyTorch's CPU wheel index.
    /path/to/capture-env/bin/python examples/capture_diagnostic_evidence.py --output DIR

The scalar objectives are deliberately simple, controlled counterexamples,
not realistic model benchmarks. No data or model downloads are needed.
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


def loss_history(lr: float) -> list[dict[str, float | int]]:
    weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=lr)
    rows = []
    for step in range(1001):
        optimizer.zero_grad()
        loss = weight.square() / 2
        rows.append({"step": step, "loss": float(loss.detach())})
        if step < 1000:
            loss.backward()
            optimizer.step()
    return rows


def gradient_history(max_norm: float | None) -> list[dict[str, float | int | None]]:
    weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=0.01)
    rows = []
    for step in range(100):
        optimizer.zero_grad()
        loss = (weight - 10).abs()
        loss.backward()
        assert weight.grad is not None
        before = float(weight.grad.abs())
        returned = None
        if max_norm is not None:
            returned = float(torch.nn.utils.clip_grad_norm_([weight], max_norm))
        after = float(weight.grad.abs())
        rows.append({"step": step, "loss": float(loss.detach()), "norm_before": before,
                     "norm_after": after, "returned_norm": returned})
        optimizer.step()
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("choose an empty directory to preserve earlier captures")
    torch.set_num_threads(1)
    slow = loss_history(0.005)
    fast = loss_history(0.1)
    sparse = [row for row in slow if row["step"] == 0 or row["step"] >= 900]
    captures = {"slow-dense": slow, "slow-sparse": sparse, "fast-dense": fast,
                "slow-sparse-no-step": [{"loss": row["loss"]} for row in sparse]}
    gradients = {}
    for name, threshold in (("unclipped", None), ("clipped", 0.25)):
        history = gradient_history(threshold)
        gradients[name] = {"clip_threshold": threshold, "observations": history}
        for where in ("before", "after"):
            captures[f"{name}-{where}"] = [
                {"step": row["step"], "loss": row["loss"], "grad_norm": row[f"norm_{where}"]}
                for row in history
            ]
    # A deliberately overflowing float32 objective, retained as a one-point
    # numerical counterexample. It is not presented as ordinary healthy training.
    weight = torch.tensor(100.0, requires_grad=True)
    loss = weight.exp()
    loss.backward()
    assert weight.grad is not None
    captures["overflow-one-step"] = [{"step": 0, "loss": float(loss.detach()),
                                      "grad_norm": float(weight.grad)}]
    # Boundary fixtures below are deliberately derived/injected, not recorded runs.
    captures["infinite-gradient-tail"] = [
        {"step": i, "grad_norm": float("inf") if i == 49 else 1.0} for i in range(50)
    ]
    captures["infinite-start-loss"] = [
        {"step": i, "loss": float("inf") if i == 0 else 0.04} for i in range(100)
    ]
    files = {}
    for name, rows in captures.items():
        path = args.output / f"{name}.csv"
        write_csv(path, rows)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
        "loss_experiment": {
            "objective": "w**2/2; w starts at 2; float64 SGD for 1000 updates",
            "learning_rates": {"slow": 0.005, "fast": 0.1},
            "sparse_selection": "same slow run, retain step 0 and steps 900 through 1000",
            "no_step_selection": "same sparse rows with the step column removed",
        },
        "gradient_experiment": {
            "objective": "abs(w-10); w starts at 0; float64 SGD, lr=0.01, 100 updates",
            "captures": gradients,
            "interpretation": "Unclipped and clipped pre-clip norms are identical; "
            "only measured before/after norms and the known capture procedure establish clipping.",
        },
        "overflow_one_step": "float32 exp(100), backward: recorded loss and gradient overflow",
        "injected_fixtures": ["infinite-gradient-tail.csv", "infinite-start-loss.csv"],
        "files": files,
    }
    (args.output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "torch": torch.__version__,
                      "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
