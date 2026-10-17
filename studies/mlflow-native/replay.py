"""Compare native SDK histories, real CLI diagnoses and retained input bytes.

Run with the checkout or an installed wheel's interpreter. An optional baseline
checkout records the previous CLI's refusal of these newly supported stores.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from trainspotter.readers import load_run

ROOT = Path(__file__).resolve().parents[2]


def token(value):
    return str(value) if not math.isfinite(value) else value.hex()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-root", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, before, readings = [], [], 0
    for capture in sorted((ROOT / "studies/mlflow-native/fixtures").iterdir()):
        manifest = json.loads((capture / "capture.json").read_text())
        history = capture / "histories.json.gz"
        assert (
            hashlib.sha256(history.read_bytes()).hexdigest()
            == manifest["nativeHistoryFile"]["sha256"]
        )
        oracle = json.loads(gzip.decompress(history.read_bytes()))
        for store in oracle["stores"]:
            source = (capture / store["path"]).relative_to(ROOT)
            for expected in store["runs"]:
                command = [
                    "analyze",
                    str(source),
                    "--run",
                    expected["runId"],
                    "--output",
                    "json",
                    "--fail-on",
                    "error",
                ]
                if args.before_root:
                    old_env = {**os.environ, "PYTHONPATH": str(args.before_root.resolve() / "src")}
                    previous = subprocess.run(
                        [sys.executable, "-m", "trainspotter.cli", *command],
                        cwd=ROOT,
                        env=old_env,
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    assert previous.returncode == 2
                    before.append(
                        {
                            "version": manifest["mlflow"],
                            "store": store["kind"],
                            "run": expected["name"],
                            "command": command,
                            "exitCode": previous.returncode,
                            "stderr": previous.stderr.strip(),
                        }
                    )
                run = load_run(ROOT / source, run_name=expected["runId"])
                names = {raw: name for name, raw in run.meta["metric_sources"].items()}
                count = 0
                assert set(names) == set(expected["histories"])
                for key, native in expected["histories"].items():
                    actual = [
                        (p.step, token(p.value), round(p.wall_time * 1000))
                        for p in run.get(names[key]).points
                    ]
                    wanted = [(p["step"], token(float(p["value"])), p["timestamp"]) for p in native]
                    assert sorted(actual) == sorted(wanted), (
                        manifest["mlflow"],
                        store["kind"],
                        expected["name"],
                        key,
                    )
                    count += len(actual)
                readings += count
                completed = subprocess.run(
                    [sys.executable, "-m", "trainspotter.cli", *command],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                report = json.loads(completed.stdout)
                assert report["run"]["mlflow"]["run"]["run_id"] == expected["runId"]
                assert report["run"]["step_axis"] == "logging_step"
                if expected["name"] in ("conflicting-steps", "default-step"):
                    assert completed.returncode == 2 and not report["run"]["input_complete"]
                    assert report["run"]["metrics"]["train/loss"]["analysis_excluded"]
                elif expected["name"] == "nonfinite" or (
                    expected["name"] == "divergence" and store["kind"] == "files"
                ):
                    assert completed.returncode == 1
                else:
                    assert completed.returncode == 0
                if expected["name"] == "baseline":
                    assert report["findings"] == []
                if expected["name"] == "overfitting":
                    assert any(f["title"] == "Possible overfitting" for f in report["findings"])
                row = {
                    "version": manifest["mlflow"],
                    "store": store["kind"],
                    "run": expected["name"],
                    "nativeReadings": count,
                    "nativeHistoryMismatches": 0,
                    "command": command,
                    "exitCode": completed.returncode,
                    "summary": report["summary"],
                    "findings": [
                        {k: f[k] for k in ("detector", "title", "severity")}
                        for f in report["findings"]
                    ],
                }
                if expected["name"] in ("stable-training", "slow-training"):
                    assert report["timing"]["status"] == "unavailable"
                    explicit = subprocess.run(
                        [sys.executable, "-m", "trainspotter.cli", *command, "--step-key", "step"],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    declared = json.loads(explicit.stdout)
                    assert explicit.returncode == 0 and declared["timing"]["status"] == "compared"
                    ratio = declared["timing"]["ratio"]
                    assert ratio > 2 if expected["name"] == "slow-training" else 0.9 < ratio < 1.1
                    row["declaredTrainingSteps"] = {"ratio": ratio, "timing": declared["timing"]}
                rows.append(row)
        for entry in manifest["files"]:
            assert (
                hashlib.sha256((capture / entry["file"]).read_bytes()).hexdigest()
                == entry["sha256"]
            )
    if before:
        (args.out / "before.json").write_text(
            json.dumps({"source": "1195f7b", "cases": before}, indent=2) + "\n"
        )
    receipt = {
        "scope": (
            "Actual native SDK replays; all recorded metric histories compared, plus CLI "
            "reports and gates. Not new training or an accuracy study."
        ),
        "python": sys.version.split()[0],
        "cases": rows,
        "nativeReadings": readings,
        "nativeHistoryMismatches": 0,
        "nativeStoreBytesUnchanged": True,
    }
    (args.out / "after.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": len(rows),
                "nativeReadings": readings,
                "mismatches": 0,
                "beforeRefused": len(before),
            }
        )
    )


if __name__ == "__main__":
    main()
