"""Check actual CLI reports against captured clocks using only stdlib arithmetic.

Run with uv run python scripts/verify_timing.py. Use --executable to verify an
installed wheel, --capture to check a fresh capture, and --reports to save its
JSON/HTML reports. No Trainspotter modules are imported into this verifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "stable-with-steps": ("stable", "clock", 0),
    "stable-without-steps": ("stable", "no-steps", 0),
    "slow-with-steps": ("slow", "clock", 1),
    "slow-direct": ("slow", "step_time", 1),
    "slow-throughput": ("slow", "1/throughput", 1),
    "slow-missing-time": ("slow", "missing", 0),
    "slow-broken-clock": ("slow", "broken", 0),
}


def median(values: list[float]) -> float | None:
    return statistics.median(values) if all(math.isfinite(v) and v > 0 for v in values) else None


def verify(args: argparse.Namespace) -> dict[str, object]:
    capture = json.loads((args.capture / "capture.json").read_text())
    command = ([str(args.executable)] if args.executable
               else [sys.executable, "-m", "trainspotter.cli"])
    checks = []
    if args.reports:
        args.reports.mkdir(parents=True, exist_ok=True)
    for name, (origin, mode, exit_code) in SCENARIOS.items():
        path = args.capture / f"{name}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert capture["files"][path.name]["sha256"] == digest
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        raw = capture["observations"][origin]
        assert [r["step"] for r in raw] == list(range(401))
        assert all(b["wall_time"] > a["wall_time"] for a, b in zip(raw[:-1], raw[1:], strict=True))
        logged = [r for r in raw if r["step"] % (4 if r["step"] <= 200 else 10) == 0]
        source = raw[1:] if mode in ("step_time", "1/throughput") else logged
        assert len(rows) == len(source) == capture["files"][path.name]["rows"]
        for i, (row, observed) in enumerate(zip(rows, source, strict=True)):
            assert float(row["loss"]) == observed["loss"]
            if "step" in row:
                assert int(row["step"]) == observed["step"]
            if "wall_time" in row:
                if mode == "missing" and i == 4:
                    assert row["wall_time"] == ""
                elif mode == "broken" and i == 4:
                    assert float(row["wall_time"]) == logged[3]["wall_time"] - 0.25
                else:
                    assert float(row["wall_time"]) == observed["wall_time"]
            if mode == "step_time":
                assert float(row["step_time"]) == observed["step_time"]
            if mode == "1/throughput":
                assert float(row["samples_per_second"]) == 128 / observed["step_time"]

        process = subprocess.run([*command, "analyze", str(path), "--output", "json",
                                  "--fail-on", "warning"], capture_output=True, text=True,
                                 cwd=ROOT, check=False, timeout=60)
        assert process.returncode == exit_code, (name, process.stderr, process.stdout)
        report = json.loads(process.stdout)
        timing = report["timing"]
        expected_ratio = None
        expected_windows = None
        if mode in ("no-steps", "missing"):
            assert timing["status"] == "unavailable" and timing["ratio"] is None
            assert timing["source"] is None
        else:
            if mode in ("clock", "broken"):
                steps = [int(r["step"]) for r in rows]
                stamps = [float(r["wall_time"]) for r in rows]
                values = [(b - a) / (s2 - s1) for a, b, s1, s2 in
                          zip(stamps[:-1], stamps[1:], steps[:-1], steps[1:], strict=True)]
                starts, positions = steps[:-1], steps[1:]
                assert timing["source"] == "wall_time(train/loss)"
                assert timing["axis"] == "step" and timing["unit"] == "s/step"
            else:
                values = [float(r["step_time"]) if mode == "step_time" else
                          1 / float(r["samples_per_second"]) for r in rows]
                starts, positions = None, list(range(len(values)))
                assert timing["source"] == mode
                assert timing["axis"] == "row_index"
                assert timing["unit"] == ("s/step" if mode == "step_time" else "s/unit")
            window = max(5, len(values) // 5)
            expected_windows = {}
            for label, lo, hi in [("baseline", 0, window),
                                  ("recent", len(values) - window, len(values))]:
                part = values[lo:hi]
                expected = {
                    "first_position": positions[lo], "last_position": positions[hi - 1],
                    "interval_start_step": starts[lo] if starts else None,
                    "observations": window,
                    "invalid_observations": sum(not math.isfinite(v) or v <= 0 for v in part),
                    "median_seconds": median(part),
                }
                assert timing[label] == expected, (name, label, timing[label], expected)
                expected_windows[label] = expected
            early, late = median(values[:window]), median(values[-window:])
            if early is not None and late is not None:
                expected_ratio = late / early
                assert math.isclose(timing["ratio"], expected_ratio, rel_tol=1e-13)
                assert timing["status"] == "compared"
            else:
                assert timing["ratio"] is None and timing["status"] == "unavailable"
            assert timing["observations"] == len(values)
        alarms = [f for f in report["findings"] if f["detector"] == "throughput"]
        assert len(alarms) == exit_code
        if alarms:
            assert alarms[0]["evidence"]["ratio"] == timing["ratio"]
        required = subprocess.run([*command, "analyze", str(path), "--output", "json",
                                   "--fail-on", "warning", "--require-timing"],
                                  capture_output=True, text=True, cwd=ROOT, timeout=60, check=False)
        expected_required = 2 if mode in ("no-steps", "missing", "broken") else exit_code
        assert required.returncode == expected_required, (name, required.stderr)
        assert json.loads(required.stdout) == report  # Failed availability gate retains the report.
        if args.reports:
            (args.reports / f"{name}.json").write_text(process.stdout)
            html = subprocess.run([*command, "analyze", str(path), "--output", "html", "--out",
                                   str(args.reports / f"{name}.html")], capture_output=True,
                                  text=True, cwd=ROOT, timeout=60, check=False)
            assert html.returncode == 0, html.stderr
        checks.append({"file": path.name, "sha256": digest, "readings": len(rows),
                       "timing_status": timing["status"], "source": timing["source"],
                       "expected_ratio": expected_ratio, "reported_ratio": timing["ratio"],
                       "windows": expected_windows, "gate_exit": process.returncode,
                       "require_timing_gate_exit": required.returncode,
                       "throughput_findings": len(alarms), "capture_and_oracle_equal": True})
    runtime = None
    interpreter = args.runtime_python or (sys.executable if not args.executable else None)
    if interpreter:
        metadata = subprocess.run([str(interpreter), "-c", """
import hashlib, json, platform
from pathlib import Path
import numpy
import trainspotter.timing
digest = hashlib.sha256(Path(trainspotter.timing.__file__).read_bytes()).hexdigest()
print(json.dumps({"python": platform.python_version(), "numpy": numpy.__version__,
    "timing_module_sha256": digest}))
"""], capture_output=True, text=True, check=True, timeout=60)
        runtime = json.loads(metadata.stdout)
    capture_hash = hashlib.sha256((args.capture / "capture.json").read_bytes()).hexdigest()
    return {"verified_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Controlled paced numerical training captures; real measured clocks",
            "cli": "installed executable" if args.executable else "source module",
            "verifier_python": platform.python_version(), "machine": platform.platform(),
            "runtime": runtime,
            "capture_sha256": capture_hash,
            "oracle": "Stdlib CSV, captured clocks, step differences and statistics.median",
            "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "examples/timing")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--runtime-python", type=Path,
                        help="Interpreter of --executable, for installed runtime/module metadata")
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/verification-timing.json")
    args = parser.parse_args()
    result = verify(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
