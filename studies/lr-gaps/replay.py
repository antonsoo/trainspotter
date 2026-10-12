"""Replay real CLI reports and check the controlled capture with Decimal arithmetic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_capture() -> dict:
    capture = ROOT / "captured"
    manifest = json.loads((capture / "capture.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest((capture / name).read_bytes()) == expected, name
    rows = list(csv.DictReader((capture / "optimizer.csv").open()))
    largest = 0.0
    with localcontext() as context:
        context.prec = 70
        for row in rows:
            step = int(row["step"])
            rate = Decimal("0.1") / 2 ** (int(step >= 40) + int(step >= 80))
            weight = (
                Decimal(3)
                * Decimal("0.9") ** min(step, 40)
                * Decimal("0.95") ** min(max(step - 40, 0), 40)
                * Decimal("0.975") ** max(step - 80, 0)
            )
            loss = 1 + weight * weight / 2
            assert math.isclose(float(row["learning_rate"]), float(rate), rel_tol=1e-14)
            for key, expected in [("weight", weight), ("loss", loss)]:
                error = abs(float(row[key]) - float(expected))
                largest = max(largest, error)
                assert math.isclose(float(row[key]), float(expected), rel_tol=1e-13, abs_tol=1e-14)
    complete = list(csv.DictReader((capture / "complete.csv").open()))
    for name in ["unreadable", "nan", "positive-inf", "negative-inf", "negative"]:
        changed = list(csv.DictReader((capture / f"{name}.csv").open()))
        differences = [
            (i, key)
            for i, (a, b) in enumerate(zip(complete, changed, strict=True))
            for key in a
            if a[key] != b[key]
        ]
        assert differences == [(10, "learning_rate")], (name, differences)
    return {
        "observations": len(rows),
        "decimalPrecision": 70,
        "maxAbsoluteWeightOrLossDifference": largest,
        "faultedExportsChangeOnlyStep10LearningRate": True,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expect", choices=["before", "after"], required=True)
    args = parser.parse_args()
    args.python = args.python.absolute()  # Keep venv interpreter symlinks intact.
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.source:
        environment["PYTHONPATH"] = str(args.source.resolve())
    else:
        environment.pop("PYTHONPATH", None)
    probe = subprocess.run(
        [
            str(args.python),
            "-c",
            """
import hashlib,json,platform
from pathlib import Path
import trainspotter
root=Path(trainspotter.__file__).parent
print(json.dumps({'python':platform.python_version(),'sourceHashes':{
 str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(root.rglob('*.py'))}}))
""",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
        cwd=args.artifacts,
    )
    runtime = json.loads(probe.stdout)
    oracle = verify_capture()
    cases = {
        name: (ROOT / "captured" / f"{name}.csv").read_bytes()
        for name in ["complete", "unreadable", "nan", "positive-inf", "negative-inf", "negative"]
    }
    synthetic = {
        "unknown-transition": [0.1] * 20 + ["LOGGER_ERROR"] + [0.01] * 20,
        "unknown-rise-phase": [0.1] * 20
        + [0.05] * 20
        + ["LOGGER_ERROR"]
        + [0.001] * 10
        + [0.1] * 20,
        "short-segment": [0.1] * 20 + ["LOGGER_ERROR"] + [0.1, 0.1, 0.01, 0.01, 0.01],
    }
    for name, values in synthetic.items():
        cases[name] = (
            "step,learning_rate\n" + "".join(f"{i},{value}\n" for i, value in enumerate(values))
        ).encode()
    records = []
    for name, body in cases.items():
        path = args.artifacts / f"{name}.csv"
        path.write_bytes(body)
        command = [
            str(args.python),
            "-m",
            "trainspotter.cli",
            "analyze",
            path.name,
            "--fail-on",
            "error",
            "--output",
            "json",
        ]
        result = subprocess.run(
            command,
            cwd=args.artifacts,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        report = json.loads(result.stdout)
        unreadable = name == "unreadable" or name in synthetic
        assert result.returncode == (2 if unreadable else 0), (name, result.stderr)
        assert report["summary"]["input_complete"] is not unreadable
        if unreadable:
            expected_step = (
                10 if name == "unreadable" else 40 if name == "unknown-rise-phase" else 20
            )
            assert report["run"]["metrics"]["lr"]["unreadable_steps"] == [expected_step]
        lr = [f for f in report["findings"] if f["detector"] == "lr_schedule"]
        jumps = [(f["step_start"], f["step_end"]) for f in lr if f["title"] == "LR discontinuity"]
        expected = (
            [(39, 40), (79, 80)]
            if name not in synthetic
            else [(19, 20)]
            if name == "unknown-rise-phase"
            else []
        )
        if args.expect == "before" and name != "complete":
            expected = [(9, 10)] if name == "negative" else []
        assert jumps == expected, (name, jumps, expected)
        if args.expect == "after":
            assert not any(f["title"] == "LR rose after decay" for f in lr)
            assert all(f["severity"] == "info" for f in lr)
            if name not in {"complete", *synthetic}:
                assert all(
                    f["evidence"]["segment_start_step"] == 11
                    for f in lr
                    if f["title"] == "LR discontinuity"
                )
        html = args.artifacts / f"{name}.html"
        html_result = subprocess.run(
            command[:-1] + ["html", "--out", html.name],
            cwd=args.artifacts,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert html_result.returncode == result.returncode, html_result.stderr
        (args.artifacts / f"{name}.json").write_text(result.stdout)
        records.append(
            {
                "name": name,
                "inputSha256": digest(body),
                "inputBytes": len(body),
                "controlledCapture": name not in synthetic,
                "exitCode": result.returncode,
                "stderr": result.stderr,
                "report": report,
                "lrDiscontinuities": jumps,
                "htmlSha256": digest(html.read_bytes()),
            }
        )
        print(f"{name}: exit {result.returncode}; LR discontinuities {jumps}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "expectation": args.expect,
                "runtime": runtime,
                "captureOracle": oracle,
                "records": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
