"""Capture real timestamps around a small, deliberately paced numerical training loop.

This is a controlled logging experiment, not a hardware or model benchmark.
The data is synthetic; the logistic-regression updates, losses and elapsed times
are computed on this machine. Sleep injects a known late delay in one run.

    uv run python examples/capture_timing.py --output /tmp/trainspotter-timing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def capture(*, late_delay: bool) -> list[dict[str, float | int | None]]:
    rng = np.random.default_rng(20261009)
    x = rng.normal(size=(128, 8))
    y = (x @ rng.normal(size=8) > 0).astype(float)
    weights = np.zeros(8)

    def loss() -> float:
        logits = x @ weights
        return float(np.mean(np.logaddexp(0, logits) - y * logits))

    rows: list[dict[str, float | int | None]] = [
        {"step": 0, "wall_time": time.perf_counter(), "loss": loss(), "step_time": None,
         "requested_seconds": None}
    ]
    for step in range(1, 401):
        start = time.perf_counter()
        probability = 1 / (1 + np.exp(-(x @ weights)))
        weights -= 0.2 * (x.T @ (probability - y) / len(y))
        value = loss()
        pace = 0.009 if late_delay and step > 200 else 0.003
        time.sleep(max(0, pace - (time.perf_counter() - start)))
        end = time.perf_counter()
        rows.append({"step": step, "wall_time": end, "loss": value,
                     "step_time": end - start, "requested_seconds": pace})
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | None]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def logged(rows: list[dict[str, float | int | None]]) -> list[dict[str, float | int | None]]:
    # The interval changes from 4 to 10 actual updates. Record 200 in both segments
    # only once, so neither repeated steps nor interpolation affect the comparison.
    return [r for r in rows if int(r["step"] or 0) % (4 if int(r["step"] or 0) <= 200 else 10) == 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "timing")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stable = capture(late_delay=False)
    slow = capture(late_delay=True)
    scenarios = {
        "stable-with-steps": (logged(stable), ["step", "wall_time", "loss"]),
        "stable-without-steps": (logged(stable), ["wall_time", "loss"]),
        "slow-with-steps": (logged(slow), ["step", "wall_time", "loss"]),
        "slow-direct": (slow[1:], ["loss", "step_time"]),
        "slow-throughput": ([{**r, "samples_per_second": 128 / float(r["step_time"] or 0)}
                             for r in slow[1:]], ["loss", "samples_per_second"]),
        "slow-missing-time": ([{**r, "wall_time": None} if i == 4 else r
                               for i, r in enumerate(logged(slow))],
                              ["step", "wall_time", "loss"]),
    }
    broken = [dict(r) for r in logged(slow)]
    broken[4]["wall_time"] = float(broken[3]["wall_time"] or 0) - 0.25
    scenarios["slow-broken-clock"] = (broken, ["step", "wall_time", "loss"])
    files = {}
    for name, (rows, fields) in scenarios.items():
        path = args.output / f"{name}.csv"
        write_csv(path, rows, fields)
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rows": len(rows)}
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Controlled, paced logistic-regression updates on synthetic data",
        "clock": "time.perf_counter, seconds; monotonic process clock, not Unix timestamps",
        "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.platform(), "seed": 20261009, "samples_per_update": 128,
        "features": 8, "learning_rate": 0.2, "updates_per_run": 400,
        "logging": "Every 4 updates through step 200, then every 10; initial observation at 0",
        "injections": {
            "slow": "Requested update duration changes from 0.003 to 0.009 seconds after step 200",
            "slow-missing-time": "Timestamp removed from the observation at step 16",
            "slow-broken-clock": "Step 16 timestamp set 0.25 seconds before step 12 timestamp",
        },
        "files": files, "observations": {"stable": stable, "slow": slow},
    }
    (args.output / "capture.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"directory": str(args.output), "files": files}, indent=2))


if __name__ == "__main__":
    main()
