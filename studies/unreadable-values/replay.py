"""Replay actual CLI reports for known-coordinate unreadable values and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    args.python = args.python.absolute()  # Preserve a venv interpreter symlink.
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
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
        capture_output=True,
        text=True,
    )
    runtime = json.loads(probe.stdout)
    capture = ROOT.parent / "finite-segments/captured"
    source = (capture / "observed.csv").read_bytes()
    capture_manifest = json.loads((capture / "capture.json").read_text())
    assert digest(source) == capture_manifest["files"]["observed.csv"]
    # Only the known logging-fault cells change. This is a synthetic text-export
    # fault applied to a retained controlled capture, not a newly captured run.
    derived = source.replace(b",nan", b",LOGGER_ERROR")
    assert derived.count(b"LOGGER_ERROR") == 20
    cases: dict[str, bytes] = {
        "captured-text-fault": derived,
        "observed-nan-control": source,
    }
    synthetic: dict[str, list[object]] = {
        "first-after-gap": [5.0] * 20 + [1.0] * 40 + ["unreadable"] * 10 + [4.0] + [1.0] * 40,
        "recovery-through-gap": [5.0] * 20
        + [1.0] * 40
        + ["unreadable"] * 10
        + [4.0] * 20
        + ["unreadable"] * 10
        + [1.0] * 40,
        "level-change-across-gap": [1.0] * 50 + ["unreadable"] * 10 + [3.0] * 50,
        "genuine-after-gap": [1.0] * 30
        + ["unreadable"] * 10
        + [5.0] * 20
        + [1.0] * 40
        + [4.0] * 20
        + [1.0] * 40,
        "finite-control": [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40,
    }
    for name, values in synthetic.items():
        cases[name] = ("step,loss\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values))).encode()
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
        proc = subprocess.run(
            command, text=True, capture_output=True, check=False, cwd=args.artifacts
        )
        assert proc.returncode in (0, 1, 2), proc.stderr
        report = json.loads(proc.stdout)
        assert report["run"]["source_path"] == path.name
        expected_gaps = []
        for row in body.decode().splitlines()[1:]:
            step, value = row.split(",")
            try:
                float(value)
            except ValueError:
                expected_gaps.append(int(step))
        targets = [
            f for f in report["findings"] if f["detector"] in ("spikes", "setbacks", "divergence")
        ]
        if args.verify:
            assert (
                report["run"]["metrics"]["train/loss"].get("unreadable_steps", []) == expected_gaps
            )
            assert report["run"]["skipped_values"] == len(expected_gaps)
            if expected_gaps:
                assert proc.returncode == 2 and not report["summary"]["input_complete"]
                assert not any("non-finite" in f["title"].lower() for f in targets)
            if name in (
                "captured-text-fault",
                "first-after-gap",
                "recovery-through-gap",
                "level-change-across-gap",
            ):
                assert targets == [], (name, targets)
            if name == "genuine-after-gap":
                assert [(f["title"], f["step_start"], f["step_end"]) for f in targets] == [
                    ("Setback", 100, 120)
                ]
                assert targets[0]["evidence"]["steps_lost"] == 20
                assert targets[0]["evidence"]["segment_start_step"] == 40
            if name == "observed-nan-control":
                assert proc.returncode == 1
                assert any(
                    f["title"] == "Non-finite training loss" and f["evidence"]["count"] == 20
                    for f in targets
                )
            if name == "finite-control":
                assert report["summary"]["input_complete"]
                assert any(
                    f["title"] == "Setback" and f["step_start"] == 60 and f["step_end"] == 80
                    for f in targets
                )
        html_path = args.artifacts / f"{name}.html"
        html_command = command[:-1] + ["html", "--out", str(html_path)]
        html = subprocess.run(
            html_command, text=True, capture_output=True, check=False, cwd=args.artifacts
        )
        assert html.returncode == proc.returncode, html.stderr
        records.append(
            {
                "name": name,
                "inputSha256": digest(body),
                "inputBytes": len(body),
                "expectedUnreadableSteps": expected_gaps,
                "exitCode": proc.returncode,
                "stderr": proc.stderr,
                "report": report,
                "htmlSha256": digest(html_path.read_bytes()),
            }
        )
        print(
            name,
            proc.returncode,
            [(f["title"], f["step_start"], f["step_end"]) for f in targets],
            flush=True,
        )
    result = {
        "runtime": runtime,
        "captureInputSha256": digest(source),
        "captureFault": (
            "Replace exactly the 20 recorded nan cells with LOGGER_ERROR; "
            "every other byte unchanged."
        ),
        "inputPathConvention": (
            "CLI receives the fixture basename while running in its artifact directory."
        ),
        "records": records,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
