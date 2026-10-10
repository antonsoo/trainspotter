"""Reproduce CLI latency, peak process memory, and findings on long synthetic logs.

Run sequentially on an otherwise quiet Linux machine. Both revisions use the
same Python and NumPy. Keep the generated inputs/reports outside the checkout.
The large input is generated from formulas here, not vendored as megabytes of CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_CLI = """
import json, resource, sys
from pathlib import Path
metrics = Path(sys.argv.pop(1))
sys.argv[0] = 'trainspotter'
try:
    from trainspotter.cli import main
    raise SystemExit(main())
finally:
    import numpy
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics.write_text(json.dumps({'peak_rss_kib': peak, 'numpy': numpy.__version__}))
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(path: Path, kind: str, n: int) -> dict[str, Any]:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if kind == "cadence":
            writer.writerow(["_step", "_runtime", "trainer/global_step", "train/loss"])
            for i in range(n):
                step = i * 2 if i < n // 2 else n + (i - n // 2) * 20
                writer.writerow([i, step / 10, step, 1 + 1 / (step + 10)])
        else:
            writer.writerow(["step", "train/loss", "wall_time"])
            for i in range(n):
                if i < n // 5:
                    value = 10 - 5 * i / (n // 5)
                elif i < 3 * n // 5:
                    value = 5.0
                else:
                    value = 5 - 4.5 * (i - 3 * n // 5) / (n - 3 * n // 5 - 1)
                if i == 737 * n // 1000:
                    value += 6
                writer.writerow([i, value, i / 10])
    return {"kind": kind, "rows": n, "bytes": path.stat().st_size, "sha256": digest(path)}


def invoke(source: Path, path: Path, report: Path, timeout: float, python: str) -> dict[str, Any]:
    metrics = report.with_suffix(report.suffix + ".metrics.json")
    command = [
        python,
        "-c",
        RUN_CLI,
        str(metrics),
        "analyze",
        path.name,
        "--output",
        report.suffix[1:],
        "--out",
        str(report),
    ]
    env = {**os.environ, "PYTHONPATH": str(source.resolve())}
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, cwd=path.parent, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"timeout_s": timeout}
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"CLI exited {result.returncode}: {result.stderr}")
    return {
        "elapsed_s": elapsed,
        **json.loads(metrics.read_text()),
        "report_bytes": report.stat().st_size,
        "report_sha256": digest(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--skip-million", action="store_true", help="Run only the two 100k cases")
    parser.add_argument(
        "--repeats", type=int, default=3, help="100k inputs; million-row input runs once"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "method": "sequential complete CLI processes; elapsed includes interpreter startup",
        "memory": "Linux peak RSS in KiB, includes interpreter, reader, detectors and rendering",
        "cases": [],
    }
    cases = [("cadence", 100_000), ("stall-spike", 100_000)]
    if not args.skip_million:
        cases.append(("stall-spike", 1_000_000))
    for kind, n in cases:
        path = args.artifacts / f"{kind}-{n}.csv"
        case = generate(path, kind, n)
        reports: dict[str, Any] = {}
        measurements: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
        for repeat in range(args.repeats if n == 100_000 else 1):
            for label, source in [("before", args.baseline_source), ("after", args.source)]:
                report = args.artifacts / f"{path.stem}-{label}-{repeat}.json"
                measurement = invoke(source, path, report, args.timeout, args.python)
                measurements[label].append(measurement)
                if "timeout_s" not in measurement:
                    reports[label] = json.loads(report.read_text())
                print(f"{path.stem} {label} {repeat + 1}: {measurement}", flush=True)
        current = reports["after"]
        if kind == "cadence":
            assert current["findings"] == []
        else:
            found = {f["detector"]: [f["step_start"], f["step_end"]] for f in current["findings"]}
            assert found == {"spikes": [737 * n // 1000] * 2, "plateau": [n // 5, 3 * n // 5]}
        equal = reports.get("before") == current if "before" in reports else None
        assert equal is not False, "Full diagnostic JSON changed"
        case.update(
            measurements=measurements,
            identical_json=equal,
            finding_ranges=[
                [f["detector"], f["step_start"], f["step_end"]] for f in current["findings"]
            ],
        )
        if all("elapsed_s" in m for samples in measurements.values() for m in samples):
            case["median_speedup"] = statistics.median(
                m["elapsed_s"] for m in measurements["before"]
            ) / statistics.median(m["elapsed_s"] for m in measurements["after"])
        if n == 1_000_000:
            report = args.artifacts / "million.html"
            case["html_after"] = invoke(args.source, path, report, args.timeout, args.python)
        evidence["cases"].append(case)
        args.out.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
