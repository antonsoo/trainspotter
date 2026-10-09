"""Verify captured predictions and actual CLI findings without importing Trainspotter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "examples/evaluation/captured")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "capture.json").read_text())
    tables = {}
    for filename, metadata in manifest["files"].items():
        path = args.capture / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"], filename
        with path.open(newline="") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
        assert len(tables[path.stem]) == metadata["rows"]
    scores = defaultdict(list)
    last_eval = {}
    for row in tables["predictions"]:
        error = (float(row["prediction"]) - float(row["target"])) ** 2
        scores[int(row["step"]), row["cohort"]].append(error)
        if row["cohort"] == "eval" and int(row["step"]) == 300:
            last_eval[int(row["example"])] = error
    max_error = 0.0
    for (step, cohort), values in scores.items():
        expected = math.fsum(values) / len(values)
        actual = float(
            tables["regression-full"][step]["loss" if cohort == "train" else "eval_loss"]
        )
        max_error = max(max_error, abs(actual - expected))
        assert math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-14)
    full = math.fsum(last_eval.values()) / len(last_eval)
    hard_indices = manifest["hard_example_indices"]
    assert set(hard_indices) == set(sorted(last_eval, key=last_eval.__getitem__, reverse=True)[:32])
    hard = math.fsum(last_eval[i] for i in hard_indices) / len(hard_indices)
    for name in ("fixed-repeat", "fixed-cohort-switch"):
        for row in tables[name]:
            expected = hard if name.endswith("switch") and int(row["step"]) % 2 else full
            assert math.isclose(float(row["eval_loss"]), expected, rel_tol=1e-13)
    # Known experimental controls, independent of the detector's implementation.
    six = [float(r["eval_loss"]) for r in tables["smooth-six"]]
    assert all(a > b for a, b in zip(six, six[1:], strict=False))
    assert len({r["eval_loss"] for r in tables["fixed-repeat"]}) == 1
    late = tables["late-training"]
    assert max(int(r["step"]) for r in late if r["eval_loss"]) == 60
    assert min(int(r["step"]) for r in late if r["loss"]) == 100
    command = (
        [str(args.executable)] if args.executable else [sys.executable, "-m", "trainspotter.cli"]
    )
    if args.reports:
        args.reports.mkdir(parents=True, exist_ok=True)
    findings = {}
    for name in tables:
        if name == "predictions":
            continue
        path = args.capture / f"{name}.csv"
        process = subprocess.run(
            [*command, "analyze", path.name, "--output", "json"],
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
            cwd=args.capture,
        )
        report = json.loads(process.stdout)
        assert report["run"]["input_complete"]
        selected = [f for f in report["findings"] if f["detector"] in ("eval_noise", "overfitting")]
        findings[name] = selected
        by_id = {f["detector"]: f for f in selected}
        if name == "fixed-repeat":
            assert selected == []
        if name == "smooth-six":
            assert bool(by_id.get("eval_noise")) == args.baseline
        if name == "late-training":
            assert by_id["overfitting"]["step_end"] == 60
            assert by_id["overfitting"]["title"] == (
                "Overfitting onset" if args.baseline else "Evaluation loss rising"
            )
        if name == "regression-full":
            assert by_id["overfitting"]["title"] == (
                "Overfitting onset" if args.baseline else "Possible overfitting"
            )
            assert by_id["overfitting"]["step_start"] == 7
        if name == "fixed-cohort-switch":
            assert "eval_noise" in by_id
            if not args.baseline:
                assert by_id["eval_noise"]["evidence"]["direction_changes"] == 18
                assert "does not measure evaluation uncertainty" in by_id["eval_noise"]["message"]
        if name == "nan-gap" and not args.baseline:
            assert by_id["overfitting"]["step_start"] == 101
            assert "eval_noise" not in by_id
        if name == "nan-final" and not args.baseline:
            assert selected == []
            assert any(f["title"] == "Non-finite evaluation values" for f in report["findings"])
        if not args.baseline:
            for f in selected:
                assert all(
                    not isinstance(v, float) or math.isfinite(v) for v in f["evidence"].values()
                )
                assert "nan%" not in f["message"]
                if f["title"] == "Possible overfitting":
                    ev = f["evidence"]
                    assert ev["training_step_start"] >= f["step_start"]
                    assert ev["training_step_end"] <= f["step_end"]
        if args.reports:
            (args.reports / f"{name}.json").write_text(process.stdout)
            subprocess.run(
                [
                    *command,
                    "analyze",
                    path.name,
                    "--output",
                    "html",
                    "--out",
                    str(args.reports / f"{name}.html"),
                ],
                check=True,
                capture_output=True,
                cwd=args.capture,
            )
    result = {
        "baseline": args.baseline,
        "captured_means_checked": len(scores),
        "max_mean_absolute_error": max_error,
        "fixed_evaluations_checked": 40,
        "files": manifest["files"],
        "findings": findings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "findings")}))


if __name__ == "__main__":
    main()
