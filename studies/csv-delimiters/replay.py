"""Replay retained exports through real readers, CLI gates, watch and HTML.

Use --source to select a checkout's src, or --python for an installed wheel.
The numeric oracle is computed by capture.py without Trainspotter imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expect", choices=["before", "after"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.python = args.python.absolute()
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if args.source:
        environment["PYTHONPATH"] = str(args.source.resolve())
    capture = json.loads((STUDY / "capture.json").read_text())

    def invoke(case, command="analyze", output="json"):
        options = ["--once", "--no-color"] if command == "watch" else ["--output", output]
        if output == "html":
            options += [
                "--out",
                str((args.out / (case["name"] + ".html")).resolve()),
                "--title",
                "CSV delimiter replay: " + case["name"],
            ]
        result = subprocess.run(
            [
                str(args.python),
                "-m",
                "trainspotter.cli",
                command,
                case["file"],
                "--fail-on",
                "error",
                *options,
            ],
            cwd=STUDY,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    records = []
    for case in capture["cases"]:
        source = STUDY / case["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        result = invoke(case)
        row = {
            "name": case["name"],
            "inputSha256": case["sha256"],
            "exitCode": result.returncode,
            "stderr": result.stderr.strip(),
        }
        if result.returncode == 1:
            report = json.loads(result.stdout)
            assert report["run"]["input_complete"]
            assert report["run"]["step_column"] == "global_step"
            assert set(report["run"]["metrics"]) == set(case["expected"])
            assert any(f["detector"] == "divergence" for f in report["findings"])
            if case["kind"] == "chart":
                assert report["run"]["chart_identity"]["selected_run"] == case["label"]
            # Read every retained numeric observation, using the selected
            # interpreter/package. Reports alone expose only summary counts.
            code = """
import hashlib, json, math, sys
from trainspotter.readers import load_run
run = load_run(sys.argv[1])
result = {}
for name, series in run.metrics.items():
    values = [[p.step, p.value.hex() if math.isfinite(p.value) else str(p.value)]
              for p in series.points]
    digest = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    result[name] = {"count": len(values), "sha256": digest}
print(json.dumps(result))
"""
            points = subprocess.run(
                [str(args.python), "-c", code, str(source.resolve())],
                cwd=args.out,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert points.returncode == 0, points.stderr
            oracle = json.loads(points.stdout)
            assert oracle == case["expected"], (case["name"], oracle, case["expected"])
            row["observations"] = oracle
            row["findings"] = report["findings"]
            if records:
                assert report["findings"] == records[0]["findings"]
                assert report["run"]["metrics"] == records[0]["metrics"]
            row["metrics"] = report["run"]["metrics"]
            if args.expect == "after":
                html = invoke(case, output="html")
                assert html.returncode == 1, html.stderr
                row["htmlSha256"] = hashlib.sha256(
                    (args.out / (case["name"] + ".html")).read_bytes()
                ).hexdigest()
        else:
            assert args.expect == "before" and case["name"] not in ("control", "chart-plain"), row
            assert result.returncode == 2 and not result.stdout, row
        watch = invoke(case, command="watch")
        assert watch.returncode == result.returncode, watch.stderr
        row["watch"] = {
            "exitCode": watch.returncode,
            "stdout": watch.stdout,
            "stderr": watch.stderr.strip(),
        }
        records.append(row)
    receipt = {
        "expectation": args.expect,
        "cases": records,
        "python": str(args.python),
        "source": str(args.source) if args.source else None,
    }
    (args.out / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "expectation": args.expect,
                "inputs": len(records),
                "diagnosed": sum(r["exitCode"] == 1 for r in records),
                "refused": sum(r["exitCode"] == 2 for r in records),
            }
        )
    )


if __name__ == "__main__":
    main()
