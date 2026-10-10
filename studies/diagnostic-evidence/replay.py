"""Replay retained captures with any installed/source trainspotter implementation.

    python studies/diagnostic-evidence/replay.py --output /tmp/replay.json [--check]
Use PYTHONPATH=/path/to/old/src to record a baseline without modifying captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report.json_report import to_json_dict

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / "examples/diagnostic-evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((CAPTURES / "capture.json").read_text())
    reports = {}
    for filename, digest in manifest["files"].items():
        path = CAPTURES / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
        run = load_run(path)
        report = to_json_dict(run, run_all(run))
        report["run"]["source_path"] = filename
        reports[path.stem] = report
    # A closed-form trajectory checks the actual optimizer capture independently.
    errors = {}
    for name, lr in (("slow", 0.005), ("fast", 0.1)):
        series = load_run(CAPTURES / f"{name}-dense.csv").metrics["train/loss"]
        expected = [2 * (1 - lr) ** (2 * point.step) for point in series.points]
        errors[name] = max(
            abs(point.value - value) for point, value in zip(series.points, expected, strict=True)
        )
        assert all(
            math.isclose(point.value, value, rel_tol=1e-10, abs_tol=1e-100)
            for point, value in zip(series.points, expected, strict=True)
        )
    dense = {
        p.step: p.value for p in load_run(CAPTURES / "slow-dense.csv").metrics["train/loss"].points
    }
    sparse = load_run(CAPTURES / "slow-sparse.csv").metrics["train/loss"].points
    assert all(dense[p.step] == p.value for p in sparse)
    gradient_captures = manifest["gradient_experiment"]["captures"]
    unclipped = gradient_captures["unclipped"]["observations"]
    clipped = gradient_captures["clipped"]["observations"]
    assert [row["norm_before"] for row in unclipped] == [row["norm_before"] for row in clipped]
    assert all(row["norm_before"] == row["norm_after"] and row["returned_norm"] is None
               for row in unclipped)
    assert all(
        row["norm_after"] < 0.25 and row["norm_before"] == row["returned_norm"] for row in clipped
    )
    evidence = {
        "torch": manifest["torch"],
        "closed_form_max_absolute_error": errors,
        "sparse_points_identical_to_dense": True,
        "identical_pre_clip_histories": True,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if args.check:
        for name in ("slow-dense", "slow-sparse", "slow-sparse-no-step", "infinite-start-loss"):
            assert not any(f["detector"] == "loss_floor" for f in reports[name]["findings"]), name
        assert any(f["detector"] == "loss_floor" for f in reports["fast-dense"]["findings"])
        for name in ("overflow-one-step", "infinite-gradient-tail"):
            norms = [f for f in reports[name]["findings"] if f["detector"] == "grad_norm"]
            assert len(norms) == 1 and norms[0]["title"] == "Non-finite gradient norm", name
            assert norms[0]["evidence"]["count"] == 1
        for name in ("clipped-before", "clipped-after", "unclipped-before", "unclipped-after"):
            norms = [f for f in reports[name]["findings"] if f["detector"] == "grad_norm"]
            assert (
                len(norms) == 1
                and norms[0]["title"] == "Gradient norms concentrated near their maximum"
            )
            assert norms[0]["evidence"]["clipping_status"] == "unknown"
    print(
        json.dumps(
            {
                name: [
                    {"detector": f["detector"], "title": f["title"]}
                    for f in report["findings"]
                    if f["detector"] in ("loss_floor", "grad_norm")
                ]
                for name, report in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
