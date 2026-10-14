"""Replay an unmodified native TensorFlow scalar capture through library and CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="omit only for an installed wheel")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observe-only", action="store_true")
    args = parser.parse_args()
    environment = dict(os.environ)
    if args.src:
        sys.path.insert(0, str(args.src.resolve()))
        environment["PYTHONPATH"] = str(args.src.resolve())
    else:
        environment.pop("PYTHONPATH", None)
    import trainspotter
    from trainspotter.detectors import run_all
    from trainspotter.readers.tensorboard import read_tensorboard
    from trainspotter.report import render_html, to_json_dict

    origin = Path(trainspotter.__file__).resolve()
    assert (
        origin.is_relative_to(args.src.resolve()) if args.src else "site-packages" in origin.parts
    )
    native = STUDY / "fixtures/events.out.tfevents.native"
    oracle = json.loads((STUDY / "results/native-capture.json").read_text())
    assert hashlib.sha256(native.read_bytes()).hexdigest() == oracle["event_sha256"]
    expected = oracle["expected"]
    observations = []
    reports = []
    for label, path in [("native-file", native), ("native-directory", native.parent)]:
        run = read_tensorboard(path)
        report = to_json_dict(run, run_all(run))
        values = [{"step": p.step, "value": p.value} for p in run.get("train/loss").points]
        controls = [{"step": p.step, "value": p.value} for p in run.get("control/loss").points]
        observations.append(
            {
                "case": label,
                "values": values,
                "controls": controls,
                "skipped_events": run.meta["skipped_events"],
                "input_complete": run.input_complete(),
                "agrees": values == controls == expected and run.input_complete(),
            }
        )
        report["run"]["source_path"] = f"<fixture:{label}>"
        reports.append(report)
    run = read_tensorboard(native)
    html = render_html(run, run_all(run), title="Synthetic native bfloat16 scalars")
    observations.append(
        {
            "case": "library-html",
            "bytes": len(html.encode()),
            "sha256": hashlib.sha256(html.encode()).hexdigest(),
            "agrees": all(p.value is not None for p in run.get("train/loss").points)
            and run.input_complete(),
        }
    )
    with tempfile.TemporaryDirectory(prefix="trainspotter-bfloat16-replay-") as scratch:
        for case, arguments in [
            ("cli-json", ["analyze", str(native), "--output", "json"]),
            ("cli-gate", ["analyze", str(native), "--output", "json", "--fail-on", "warning"]),
            ("watch-gate", ["watch", str(native), "--once", "--no-color", "--fail-on", "warning"]),
        ]:
            result = subprocess.run(
                [sys.executable, "-m", "trainspotter.cli", *arguments, "--format", "tensorboard"],
                cwd=scratch,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            body = (
                json.loads(result.stdout)
                if case != "watch-gate" and result.stdout.lstrip().startswith("{")
                else None
            )
            agrees = result.returncode == 0
            if body is not None:
                agrees = agrees and body["run"]["input_complete"]
                agrees = agrees and body["run"]["metrics"]["train/loss"]["count"] == 5
            elif case == "watch-gate":
                agrees = agrees and "unreadable" not in result.stdout
            else:
                agrees = False
            observations.append(
                {
                    "case": case,
                    "exit": result.returncode,
                    "stdout": result.stdout.replace(str(native), "<native-event-file>"),
                    "stderr": result.stderr.replace(str(native), "<native-event-file>"),
                    "agrees": agrees,
                }
            )
    result = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "tensorboard": importlib.metadata.version("tensorboard"),
        "source_sha256": hashlib.sha256(
            (origin.parent / "readers/tensorboard.py").read_bytes()
        ).hexdigest(),
        "population": (
            "Unmodified native TensorFlow 2.20 summary bytes; "
            "synthetic constants, no model training."
        ),
        "event_sha256": oracle["event_sha256"],
        "count": len(observations),
        "disagreements": sum(not item["agrees"] for item in observations),
        "observations": observations,
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"count": result["count"], "disagreements": result["disagreements"]}))
    if result["disagreements"] and not args.observe_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
