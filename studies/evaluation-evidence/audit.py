"""Compare evaluation detectors on the pinned public logs using identical readers.

No new collection and no labels of true overfitting or measurement noise. The
study measures changed observations and the support for training attribution.
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

from trainspotter.detectors.eval_noise import EvalNoiseDetector
from trainspotter.detectors.overfitting import OverfittingDetector
from trainspotter.readers import load_run

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / "studies/hub-census"
MODULES = {"eval_noise": EvalNoiseDetector, "overfitting": OverfittingDetector}


def shape(findings: list[dict]) -> list[tuple]:
    # A more cautious title and new evidence fields affect every retained
    # finding. Count structural changes separately from that wording change.
    return [
        (
            f["detector"],
            f["severity"],
            f["step_start"],
            f["step_end"],
            f["title"] in ("Overfitting onset", "Possible overfitting"),
        )
        for f in findings
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="76954b0")
    parser.add_argument("--cache", type=Path, default=HUB / "cache")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline_ref], text=True).strip()
    old, new, hashes = {}, {}, {}
    for name, cls in MODULES.items():
        path = f"src/trainspotter/detectors/{name}.py"
        source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
        module = types.ModuleType(f"trainspotter.detectors.{name}_baseline")
        module.__package__ = "trainspotter.detectors"
        exec(compile(source, f"{name}_baseline.py", "exec"), module.__dict__)
        old[name], new[name] = getattr(module, cls.__name__)(), cls()
        hashes[name] = {
            "before": hashlib.sha256(source.encode()).hexdigest(),
            "after": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
    slices = {}
    started = time.monotonic()
    for filename in ("manifest.json", "manifest-heldout.json"):
        counts: Counter[str] = Counter()
        changes, unavailable = [], []
        manifest_bytes = (HUB / filename).read_bytes()
        for row in json.loads(manifest_bytes)["logs"]:
            counts["listed"] += 1
            key = f"{row['repo']}@{row['revision']}:{row['path']}"
            path = args.cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")
            try:
                run = load_run(path, fmt="hf").analysis_view()
            except (OSError, ValueError) as exc:
                unavailable.append({"source": key, "reason": type(exc).__name__})
                counts["unavailable"] += 1
                continue
            counts["read"] += 1
            evaluation = run.get("eval/loss")
            if evaluation is None:
                counts["no_usable_eval"] += 1
                continue
            counts["with_eval"] += 1
            before, after = [], []
            for name in MODULES:
                previous = [asdict(f) for f in old[name].run(run)]
                current = [asdict(f) for f in new[name].run(run)]
                counts[f"{name}_before"] += len(previous)
                counts[f"{name}_after"] += len(current)
                counts[f"{name}_removed_runs"] += bool(previous) and not current
                counts[f"{name}_added_runs"] += bool(current) and not previous
                before.extend(previous)
                after.extend(current)
            had_training = any(f["title"] == "Overfitting onset" for f in before)
            has_training = any(f["title"] == "Possible overfitting" for f in after)
            counts["training_attribution_before"] += had_training
            counts["training_attribution_after"] += has_training
            counts["training_attribution_removed"] += had_training and not has_training
            counts["training_attribution_added"] += has_training and not had_training
            if shape(before) == shape(after):
                continue
            counts["structurally_changed_runs"] += 1
            training = run.get("train/loss")
            changes.append(
                {
                    "source": key,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "before": before,
                    "after": after,
                    "evaluation_observations": len(evaluation),
                    "evaluation_span": [evaluation.points[0].step, evaluation.points[-1].step],
                    "training_span": [training.points[0].step, training.points[-1].step]
                    if training is not None and len(training)
                    else None,
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
        "detector_sha256": hashes,
        "scope": "Two evaluation detectors only, identical current readers; no causal ground truth",
        "seconds": round(time.monotonic() - started, 3),
        "slices": slices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Baseline evidence can contain NaN: encode explicitly, never emit invalid JSON.
    def sanitize(value: object) -> object:
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            return str(value)
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value

    args.out.write_text(json.dumps(sanitize(result), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
