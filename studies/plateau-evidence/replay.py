"""Capture real CLI output and independently check each reported median-level bound."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def plateau_witness(path: Path, finding: dict) -> None:
    with path.open() as handle:
        readings = [(int(row["step"]), float(row["loss"])) for row in csv.DictReader(handle)
                    if row["loss"]]
    spans: list[list[tuple[int, float]]] = [[]]
    for reading in readings:
        if math.isfinite(reading[1]):
            spans[-1].append(reading)
        else:
            spans.append([])
    segment = next(span for span in spans if span and span[0][0] <= finding["step_start"]
                   and span[-1][0] >= finding["step_end"])
    evidence = finding["evidence"]
    values = [v for _, v in segment]
    half = evidence["smoothing_window"] // 2
    # Sorted standard-library medians, not the detector's vectorized algorithm.
    smooth = [statistics.median(values[max(0, i - half):i + half + 1])
              for i in range(len(values))]
    selected = [smooth[i] for i, (step, _) in enumerate(segment)
                if finding["step_start"] <= step <= finding["step_end"]]
    low, high = min(selected), max(selected)
    assert (low, high) == (evidence["level_min"], evidence["level_max"])
    bound = max(evidence["rel_change_threshold"] * max(abs(low), abs(high), 1e-12),
                evidence["noise_allowance"])
    assert high - low < bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    decay = [5 * .9**i + 5 for i in range(40)]
    later = [5 * .9**i + .5 for i in range(40)]
    examples = {
        "returning-rise": [1.0] * 50 + [4.0] * 20 + [1.0] * 50 + [.1] * 40,
        "symmetric-excursion": [1.0] * 20 + [4.0] * 20 + [1.0] * 20,
        "after-gap": [1.0] * 30 + [math.nan] * 10 + [5.0] * 20 + [1.0] * 40
                     + [4.0] * 20 + [1.0] * 40,
        "minimum-window": [5.0] * 30,
        "genuine-stall": decay + [5 + .001 * (-1)**i for i in range(60)] + later,
        "brief-blip": decay + [5 + .001 * (-1)**i + (.5 if 28 <= i < 33 else 0)
                               for i in range(60)] + later,
        "converged-floor": [5 * .9**i + 1 for i in range(40)]
                           + [1.001, .999, 1, 1.002, .998] * 12,
        "future-lr": [5.0] * 100,
        "invalid-lr": [None] * 50 + [5.0] * 100,
    }
    paths = {}
    for name, values in examples.items():
        path = args.artifacts / f"{name}.csv"
        rates = ({80: .05, 90: 1} if name == "future-lr" else
                 {0: 1, 20: math.inf, 40: .05} if name == "invalid-lr" else {})
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["step", "loss", "lr"])
            writer.writerows((i, value, rates.get(i, "")) for i, value in enumerate(values)
                             if value is not None or i in rates)
        paths[name] = path
    for row in json.loads((ROOT / "public/manifest.json").read_text()):
        path = ROOT / "public" / row["derived_csv"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["derived_sha256"]
        paths[path.stem] = path
    command = (
        [str(args.executable)] if args.executable else [sys.executable, "-m", "trainspotter.cli"]
    )
    cases = {}
    expected_counts = {"returning-rise": 2, "symmetric-excursion": 0, "after-gap": 1,
                       "minimum-window": 1, "genuine-stall": 1, "brief-blip": 1,
                       "converged-floor": 0, "future-lr": 1, "invalid-lr": 1,
                       "public-omni": 1, "public-rca": 5}
    for name, path in paths.items():
        result = subprocess.run([*command, "analyze", str(path.resolve()), "--output", "json",
                                 "--fail-on", "error"], capture_output=True, text=True, check=False)
        assert not result.stderr, (name, result.stderr)
        report = json.loads(result.stdout)
        assert result.returncode in (0, 1), (name, result.returncode)
        plateau = [f for f in report["findings"] if f["detector"] == "plateau"]
        if args.verify:
            assert len(plateau) == expected_counts[name], (name, plateau)
            for finding in plateau:
                plateau_witness(path, finding)
            if name == "returning-rise":
                assert [(f["step_start"], f["step_end"]) for f in plateau] == [(0, 49), (70, 119)]
        cases[name] = {
            "provenance": "derived public step/loss data; see public/manifest.json"
                          if name.startswith("public-") else "synthetic boundary/control fixture",
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exit_code": result.returncode, "input_complete": report["run"]["input_complete"],
            "findings": report["findings"],
        }
    args.out.write_text(json.dumps({"independent_level_bounds_checked": args.verify,
                                   "cases": cases}, indent=2) + "\n")
    print(json.dumps({name: [(f["step_start"], f["step_end"])
                            for f in row["findings"] if f["detector"] == "plateau"]
                      for name, row in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
