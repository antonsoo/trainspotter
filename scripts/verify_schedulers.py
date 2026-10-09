"""Replay captured PyTorch runs through the actual CLI, without importing Trainspotter.

Verify rates against closed-form schedules and reports against known experiment
boundaries. --baseline checks the retained before-version's missing rebounds;
--executable checks an installed wheel. See docs/learning-rate-rebounds.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Prior peak, last low, final high. These follow the capture's configuration,
# not a second implementation of the production reversal detector.
EPISODES = {
    "linear-resume-reset": [(40, 201, 239)],
    "linear-resume-reset-sparse": [(40, 200, 230)],
    "cosine-overrun": [(0, 120, 239)],
    "cosine-overrun-sparse": [(0, 120, 230)],
    "cosine-restarts": [(0, 79, 80), (80, 159, 160)],
}


def expected_rate(name: str, step: int) -> float:
    if name.startswith("linear"):
        t = step - 200 if "reset" in name and step > 200 else step
        return 0.1 * min(t / 40, max(0.0, (240 - t) / 200))
    if name.startswith("cosine"):
        t, horizon = (
            (step % 80, 80) if "restarts" in name else (step, 120 if "overrun" in name else 240)
        )
        return 0.05 * (1 + math.cos(math.pi * t / horizon))
    if name == "step-decay-control":
        return 0.1 * 0.1 ** (step // 80)
    if name == "one-cycle-control":
        if step <= 59:
            return 0.004 + (0.1 - 0.004) / 2 * (1 - math.cos(math.pi * step / 59))
        return 0.0000004 + (0.1 - 0.0000004) / 2 * (1 + math.cos(math.pi * (step - 59) / 180))
    raise ValueError(name)


def verify(args: argparse.Namespace) -> dict[str, object]:
    capture = json.loads((args.capture / "capture.json").read_text())
    command = (
        [str(args.executable)] if args.executable else [sys.executable, "-m", "trainspotter.cli"]
    )
    if args.reports:
        args.reports.mkdir(parents=True, exist_ok=True)
    tables = {}
    checks = {}
    for filename, provenance in sorted(capture["files"].items()):
        path = args.capture / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == provenance["sha256"], filename
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        tables[path.stem] = rows
        assert len(rows) == provenance["rows"]
        steps = [int(r["step"]) for r in rows]
        stride = provenance.get("every_nth_update", 1)
        assert steps == list(range(0, 240, stride))
        rates = {int(r["step"]): float(r["learning_rate"]) for r in rows}
        max_error = 0.0
        for step, rate in rates.items():
            expected = expected_rate(path.stem, step)
            assert math.isclose(rate, expected, rel_tol=1e-12, abs_tol=1e-15), (filename, step)
            max_error = max(max_error, abs(rate - expected))
        process = subprocess.run(
            [*command, "analyze", str(path), "--output", "json", "--fail-on", "warning"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            cwd=ROOT,
        )
        assert process.returncode == 0, (filename, process.returncode, process.stderr)
        report = json.loads(process.stdout)
        assert report["run"]["input_complete"]
        findings = [f for f in report["findings"] if f["detector"] == "lr_schedule"]
        rebounds = [f for f in findings if f["title"] == "LR rose after decay"]
        expected_episodes = [] if args.baseline else EPISODES.get(path.stem, [])
        assert len(rebounds) == len(expected_episodes), (filename, findings)
        for finding, (peak, low, high) in zip(rebounds, expected_episodes, strict=True):
            assert finding["severity"] == "info"
            assert (finding["step_start"], finding["step_end"]) == (low, high)
            evidence = finding["evidence"]
            assert evidence["peak_step"] == peak
            assert evidence["low_step"] == low and evidence["high_step"] == high
            assert evidence["earlier_peak"] == rates[peak]
            assert evidence["low_value"] == rates[low] and evidence["high_value"] == rates[high]
            # The independently specified low/high bounds constrain the first
            # crossing; this is not a search for turns in an arbitrary signal.
            confirmed = next(
                s
                for s in steps
                if low < s <= high and (rates[s] - rates[low] > 0.01 * max(rates[peak], rates[s]))
            )
            assert evidence["confirmed_step"] == confirmed
            assert "log alone does not establish a scheduler fault" in finding["message"]
        if not args.baseline and path.stem == "cosine-restarts":
            assert not any(f["title"] == "LR discontinuity" for f in findings)
        if args.reports:
            (args.reports / f"{path.stem}.json").write_text(process.stdout)
            subprocess.run(
                [
                    *command,
                    "analyze",
                    str(path),
                    "--output",
                    "html",
                    "--out",
                    str(args.reports / f"{path.stem}.html"),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
                cwd=ROOT,
            )
        checks[path.stem] = {
            "sha256": digest,
            "observations": len(rows),
            "maximum_rate_oracle_error": max_error,
            "exit_code": process.returncode,
            "lr_findings": findings,
        }
    assert tables["linear-control"] == tables["linear-resume-restored"]
    for filename, provenance in capture["files"].items():
        if "derived_from" in provenance:
            full = tables[provenance["derived_from"]]
            assert tables[Path(filename).stem] == full[:: provenance["every_nth_update"]]
    for name, policy in [("linear-resume-restored", 200), ("linear-resume-reset", 0)]:
        resume = capture["scenarios"][name]["resume"]
        assert resume["checkpoint_scheduler_epoch"] == 200
        assert resume["resumed_scheduler_epoch"] == policy
        assert math.isclose(resume["optimizer_lr"], 0.02, rel_tol=1e-12)
    return {
        "label": args.label,
        "baseline": args.baseline,
        "torch": capture["torch"],
        "correct_resume_matches_uninterrupted": True,
        "scenarios": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "examples/schedulers/captured")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--label", default="current")
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded if args.out is None else f"Verified 11 captured CSVs; wrote {args.out}")


if __name__ == "__main__":
    main()
