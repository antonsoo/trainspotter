"""Replay two changed detectors over locally cached, revision-addressed HF logs.

This measures compatibility and changed observations, not diagnostic accuracy.
    python studies/diagnostic-evidence/audit.py --output /tmp/audit.json
No downloads or training jobs occur. Missing/invalid inputs remain listed.
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

import numpy as np

from trainspotter.detectors.grad_norm import GradNormDetector
from trainspotter.detectors.loss_floor import LossFloorDetector
from trainspotter.readers import load_run
from trainspotter.report.json_report import _json_safe

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / "studies/hub-census"
CLASSES = {"grad_norm": GradNormDetector, "loss_floor": LossFloorDetector}


def structure(findings):
    # The concentration title changes for every matching run. Count that
    # editorial correction separately from changes in the observed evidence.
    return [
        {
            "kind": "concentration" if "ceiling" in f["evidence"] else f["title"],
            "span": [f["step_start"], f["step_end"]],
            "severity": f["severity"],
            "count": f["evidence"].get("count"),
            "value": f["evidence"].get("value"),
            "ceiling": f["evidence"].get("ceiling"),
        }
        for f in findings
    ]


def findings_dicts(findings):
    result = []
    for finding in findings:
        data = asdict(finding)
        data["evidence"] = {key: _json_safe(value) for key, value in finding.evidence.items()}
        result.append(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="93c74bd")
    parser.add_argument("--cache", type=Path, default=HUB / "cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
    ).strip()
    before, after, sources = {}, {}, {}
    for name, cls in CLASSES.items():
        path = f"src/trainspotter/detectors/{name}.py"
        source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
        module = types.ModuleType(f"trainspotter.detectors.{name}_baseline")
        module.__package__ = "trainspotter.detectors"
        exec(compile(source, f"{name}_baseline.py", "exec"), module.__dict__)
        before[name], after[name] = getattr(module, cls.__name__)(), cls()
        sources[name] = {
            "before": hashlib.sha256(source.encode()).hexdigest(),
            "after": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
    slices = {}
    started = time.monotonic()
    with np.errstate(all="ignore"):
        for filename in ("manifest.json", "manifest-heldout.json"):
            counts = Counter()
            changed, unavailable = [], []
            fingerprint = hashlib.sha256()
            manifest = (HUB / filename).read_bytes()
            for row in json.loads(manifest)["logs"]:
                counts["listed"] += 1
                key = f"{row['repo']}@{row['revision']}:{row['path']}"
                path = args.cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")
                try:
                    raw = path.read_bytes()
                    run = load_run(path, fmt="hf")
                except (OSError, ValueError) as exc:
                    counts["unavailable"] += 1
                    unavailable.append({"source": key, "reason": type(exc).__name__})
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                fingerprint.update(f"{key}\n{digest}\n".encode())
                counts["read"] += 1
                counts["incomplete_inputs"] += not run.input_complete()
                run = run.analysis_view()
                for name in CLASSES:
                    previous = findings_dicts(before[name].run(run))
                    current = findings_dicts(after[name].run(run))
                    for label, findings in (("before", previous), ("after", current)):
                        counts[f"{name}_{label}_findings"] += len(findings)
                        counts[f"{name}_{label}_runs"] += bool(findings)
                        counts[f"{name}_{label}_concentration"] += sum(
                            "ceiling" in f["evidence"] for f in findings
                        )
                        counts[f"{name}_{label}_nonfinite"] += sum(
                            f["title"] == "Non-finite gradient norm" for f in findings
                        )
                    old_shape, new_shape = structure(previous), structure(current)
                    if old_shape != new_shape:
                        counts[f"{name}_changed_observations"] += 1
                        changed.append(
                            {
                                "source": key,
                                "sha256": digest,
                                "detector": name,
                                "before": old_shape,
                                "after": new_shape,
                            }
                        )
                if counts["read"] % 500 == 0:
                    print(filename, "read", counts["read"], flush=True)
            slices[filename] = {
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "input_fingerprint_sha256": fingerprint.hexdigest(),
                "counts": dict(counts),
                "unavailable": unavailable,
                "changed": changed,
            }
            print(filename, dict(counts), flush=True)
    result = {
        "baseline_revision": revision,
        "detector_sha256": sources,
        "scope": "Two detectors; identical current readers; no causal ground truth",
        "seconds": round(time.monotonic() - started, 3),
        "slices": slices,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
