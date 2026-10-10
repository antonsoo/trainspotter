"""Verify CLI results against retained SDK records and independent update clocks.

This verifier imports no Trainspotter modules. --executable exercises a freshly
installed wheel. --baseline-source points to an older checkout's src directory
for a before/after comparison of the exact same captured inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(rows: list[dict[str, Any]], clock: str, axis: str) -> dict[str, Any]:
    intervals = [(float(b[clock]) - float(a[clock])) / (int(b[axis]) - int(a[axis]))
                 for a, b in zip(rows[:-1], rows[1:], strict=True)]
    assert all(math.isfinite(t) and t > 0 for t in intervals)
    # These captures have 100 intervals: the report's default first and last 20%.
    assert len(intervals) == 100
    first, last = statistics.median(intervals[:20]), statistics.median(intervals[-20:])
    return {"baseline_seconds": first, "recent_seconds": last, "ratio": last / first,
            "baseline_range": [int(rows[0][axis]), int(rows[20][axis])],
            "recent_range": [int(rows[-21][axis]), int(rows[-1][axis])],
            "observations": len(intervals)}


def invoke(command: list[str], arguments: list[str], *, source: Path | None = None
           ) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    if source:
        env["PYTHONPATH"] = str(source.resolve())
    return subprocess.run(command + arguments, capture_output=True, text=True, timeout=30, env=env)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.capture / "capture.json").read_text())
    for name, sha in manifest["files"].items():
        assert digest(args.capture / name) == sha, name
    command = ([str(args.executable.resolve())] if args.executable
               else [sys.executable, "-m", "trainspotter.cli"])
    if args.reports:
        args.reports.mkdir(parents=True, exist_ok=True)
    checks = []
    clocks = {}
    baseline_results = []
    for name in ("stable", "slow"):
        records = [json.loads(line) for line in
                   (args.capture / f"{name}-history.jsonl").read_text().splitlines()]
        sdk_rows = []
        for record in records:
            row = {}
            for item in record["item"]:
                key = item.get("key") or "/".join(item["nested_key"])
                assert key not in row
                row[key] = json.loads(item["value_json"])
            sdk_rows.append(row)
        raw = manifest["captures"][name]["observations"]
        assert [r["step"] for r in raw] == list(range(1201))
        assert all(b["elapsed"] > a["elapsed"] for a, b in zip(raw[:-1], raw[1:], strict=True))
        observed = [r for r in raw if r["step"] % (4 if r["step"] <= 200 else 20) == 0]
        assert len(sdk_rows) == len(observed) == 101
        for i, (row, update) in enumerate(zip(sdk_rows, observed, strict=True)):
            assert row["_step"] == i
            assert row["trainer/global_step"] == update["step"]
            assert row["capture/elapsed"] == update["elapsed"]
            assert row["train/loss"] == update["loss"]
        sdk_clock = compare(sdk_rows, "_runtime", "trainer/global_step")
        update_clock = compare(observed, "elapsed", "step")
        logging_clock = compare(sdk_rows, "_runtime", "_step")
        assert math.isclose(sdk_clock["ratio"], update_clock["ratio"], rel_tol=0.05)
        assert math.isclose(logging_clock["ratio"], sdk_clock["ratio"] * 5, rel_tol=1e-10)
        if name == "stable":
            assert 0.8 < update_clock["ratio"] < 1.2
        else:
            assert update_clock["ratio"] > 2
        clocks[name] = {"sdk_training": sdk_clock, "independent_updates": update_clock,
                        "logging_counter": logging_clock}
        for variant, step_key in (("training", None), ("logging", None), ("custom", None),
                                  ("custom", "optimizer/updates")):
            source = args.capture / f"{name}-{variant}.csv"
            with source.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            assert len(csv_rows) == len(sdk_rows)
            for csv_row, sdk in zip(csv_rows, sdk_rows, strict=True):
                expected = dict(sdk)
                if variant == "logging":
                    del expected["trainer/global_step"]
                elif variant == "custom":
                    expected["optimizer/updates"] = expected.pop("trainer/global_step")
                assert csv_row.keys() == expected.keys()
                assert all(float(csv_row[key]) == value for key, value in expected.items())
            selection = ["--step-key", step_key] if step_key else []
            available = variant == "training" or step_key is not None
            arguments = ["analyze", str(source), *selection, "--fail-on", "warning"]
            ordinary = invoke(command, [*arguments, "--output", "json"])
            guarded = invoke(command, [*arguments, "--output", "json", "--require-timing"])
            expected_exit = int(name == "slow") if available else 0
            assert ordinary.returncode == expected_exit, ordinary.stderr
            assert guarded.returncode == (expected_exit if available else 2), guarded.stderr
            report = json.loads(ordinary.stdout)
            assert json.loads(guarded.stdout) == report
            assert report["run"]["input_complete"]
            timing = report["timing"]
            assert report["run"]["step_selection"] == ("explicit" if step_key else "automatic")
            if available:
                assert timing["status"] == "compared"
                assert timing["axis"] == report["run"]["step_axis"] == "step"
                assert math.isclose(timing["ratio"], sdk_clock["ratio"], rel_tol=1e-12)
                for window, field in (("baseline", "baseline_range"), ("recent", "recent_range")):
                    assert [timing[window]["interval_start_step"], timing[window]["last_position"]
                            ] == sdk_clock[field]
                    assert math.isclose(timing[window]["median_seconds"],
                                        sdk_clock[f"{window}_seconds"], rel_tol=1e-12)
            else:
                assert timing["status"] == "unavailable" and timing["ratio"] is None
                assert timing["axis"] == report["run"]["step_axis"] == "logging_step"
            finding = [f for f in report["findings"] if f["detector"] == "throughput"]
            assert len(finding) == int(available and name == "slow")
            slug = f"{name}-{variant}" + ("-selected" if step_key else "")
            if args.reports:
                (args.reports / f"{slug}.json").write_text(ordinary.stdout)
                rendered = invoke(command, [*arguments, "--output", "html", "--out",
                                             str(args.reports / f"{slug}.html")])
                assert rendered.returncode == expected_exit, rendered.stderr
            checks.append({"case": slug, "input_sha256": digest(source),
                           "ratio": timing["ratio"], "axis": timing["axis"],
                           "step_column": report["run"]["step_column"],
                           "ordinary_exit": ordinary.returncode,
                           "timing_gate_exit": guarded.returncode,
                           "throughput_findings": len(finding)})
            if args.baseline_source and variant in ("training", "logging"):
                before = invoke([sys.executable, "-m", "trainspotter.cli"],
                                [*arguments, "--output", "json"], source=args.baseline_source)
                assert before.returncode == 1, before.stderr
                old = json.loads(before.stdout)
                assert math.isclose(old["timing"]["ratio"], logging_clock["ratio"], rel_tol=1e-12)
                baseline_results.append({"case": slug, "ratio": old["timing"]["ratio"],
                                         "exit": before.returncode})
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier_python": platform.python_version(), "sdk_version": manifest["wandb"],
        "capture_sha256": digest(args.capture / "capture.json"),
        "command": ([args.executable.name] if args.executable
                    else ["python", "-m", "trainspotter.cli"]),
        "history_records_checked": 202, "update_observations_checked": 2402,
        "clocks": clocks, "checks": checks,
        "baseline_ref": args.baseline_ref, "baseline": baseline_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "examples/wandb-axes")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args), indent=2) + "\n"
    if args.out:
        args.out.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
