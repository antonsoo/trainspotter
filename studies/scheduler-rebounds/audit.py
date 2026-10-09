"""Measure the learning-rate check's change on the existing public-log corpus.

Requires the cached files from studies/hub-census/collect.py. Both detector
versions use the current reader; only the LR detector is loaded from the
specified baseline Git revision. Unknown schedules are never labelled faulty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from trainspotter.detectors.lr_schedule import LRScheduleDetector
from trainspotter.readers import load_run

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / "studies/hub-census"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="91bd480")
    parser.add_argument("--cache", type=Path, default=HUB / "cache")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/trainspotter/detectors/lr_schedule.py"],
        cwd=ROOT,
        text=True,
    )
    legacy = types.ModuleType("trainspotter.detectors.lr_schedule_baseline")
    legacy.__package__ = "trainspotter.detectors"
    exec(compile(source, "lr_schedule_baseline.py", "exec"), legacy.__dict__)
    old = legacy.LRScheduleDetector()
    new = LRScheduleDetector()
    slices = {}
    started = time.monotonic()
    for filename in ("manifest.json", "manifest-heldout.json"):
        counts: Counter[str] = Counter()
        changes = []
        unavailable = []
        manifest_bytes = (HUB / filename).read_bytes()
        for row in json.loads(manifest_bytes)["logs"]:
            counts["listed"] += 1
            key = f"{row['repo']}@{row['revision']}:{row['path']}"
            path = args.cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")
            try:
                run = load_run(path, fmt="hf").analysis_view()
            except (OSError, ValueError) as exc:
                counts["unavailable"] += 1
                unavailable.append({"source": key, "reason": type(exc).__name__})
                continue
            counts["read"] += 1
            lr = run.get("lr")
            if lr is None:
                counts["no_usable_lr"] += 1
                continue
            counts["with_lr"] += 1
            counts["lr_observations"] += len(lr)
            before = [asdict(f) for f in old.run(run)]
            after = [asdict(f) for f in new.run(run)]
            rebounds = [f for f in after if f["title"] == "LR rose after decay"]
            counts["before_findings"] += len(before)
            counts["after_findings"] += len(after)
            counts["rebound_episodes"] += len(rebounds)
            counts["runs_with_rebounds"] += bool(rebounds)
            assert all(f["severity"] == "info" for f in after)
            if before == after:
                continue
            counts["changed_runs"] += 1
            # Numeric source excerpts only: do not redistribute whole third-party logs.
            steps = {f["step_start"] for f in rebounds} | {f["step_end"] for f in rebounds}
            steps |= {f["evidence"]["peak_step"] for f in rebounds}
            steps |= {f["evidence"]["confirmed_step"] for f in rebounds}
            observations = [{"step": p.step, "lr": p.value} for p in lr.points if p.step in steps]
            changes.append(
                {
                    "source": key,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "before": before,
                    "after": after,
                    "lr_observations": observations,
                }
            )
        slices[filename] = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "counts": dict(counts),
            "unavailable": unavailable,
            "changes": changes,
        }
        print(filename, dict(counts), flush=True)
    result = {
        "baseline_revision": revision,
        "baseline_detector_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_detector_sha256": hashlib.sha256(
            (ROOT / "src/trainspotter/detectors/lr_schedule.py").read_bytes()
        ).hexdigest(),
        "scope": "LR detector only; identical readers; observed reversals, not proven faults",
        "seconds": round(time.monotonic() - started, 3),
        "slices": slices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
