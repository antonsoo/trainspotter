"""Exercise ordinal JSONL record loss through the actual JSON and HTML CLI.

The retained CPU experiment is projected to counter-free JSONL; faults at its
observed NaN positions are controlled transformations, not native logger output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAULTS = {
    "truncated": '{"loss":',
    "duplicate": '{"loss":NaN,"loss":999,"wall_time":123456}',
    "array": '["loss",999]',
    "scalar-null": "null",
}
TARGETS = {"divergence", "spikes", "setbacks"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(values: list[float | None], fault: str | None) -> bytes:
    return ("\n".join(
        FAULTS[fault] if value is None and fault else json.dumps({"loss": value})
        for value in values
    ) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    python = str(args.python.absolute())  # Preserve the virtualenv entry point.
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    runtime = subprocess.run([python, "-c", """
import hashlib,json,platform,numpy
from pathlib import Path
import trainspotter
root=Path(trainspotter.__file__).parent
print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,
 'sourceHashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.typed'}}}))
"""], capture_output=True, text=True, env=env, cwd=artifacts, check=True)
    capture = ROOT.parent / "finite-segments/captured"
    original = (capture / "observed.csv").read_bytes()
    manifest = json.loads((capture / "capture.json").read_text())
    assert digest(original) == manifest["files"]["observed.csv"]
    captured: list[float | None] = [
        None if row["loss"] == "nan" else float(row["loss"])
        for row in csv.DictReader(original.decode().splitlines())
    ]
    assert sum(value is None for value in captured) == 20
    patterns: dict[str, list[float | None]] = {
        "captured": captured,
        "rise": [1.] * 50 + [None] * 10 + [3.] * 50,
        "spike": [5.] * 20 + [1.] * 40 + [None] * 10 + [4.] + [1.] * 40,
        "setback": [5.] * 20 + [1.] * 40 + [None] * 10 + [4.] * 20 + [None] * 10 + [1.] * 40,
        "local": [1.] * 30 + [None] * 10 + [5.] * 20 + [1.] * 40 + [4.] * 20 + [1.] * 40,
    }
    cases: list[tuple[str, bytes, list[int], str]] = []
    for kind, values in patterns.items():
        for fault in FAULTS:
            cases.append((f"{kind}-{fault}", encode(values, fault),
                          [i for i, value in enumerate(values) if value is None], kind))
    cases.extend([
        ("finite-control", encode([5.] * 20 + [1.] * 40 + [4.] * 20 + [1.] * 40, None),
         [], "finite"),
        ("nan-control", encode([float("nan") if v is None else v for v in captured], None),
         [], "nonfinite"),
        ("sparse-control", b'{"loss":1}\n\n{ "loss":null }\n{ }\n{"loss":2}\n',
         [], "sparse"),
        ("explicit-control", b'{"step":100,"loss":1}\n{"step":150,"loss":\n'
         b'{"step":200,"loss":2}\n', [], "explicit"),
    ])
    records = []
    for name, body, gaps, kind in cases:
        path = artifacts / (name + ".jsonl")
        path.write_bytes(body)
        command = [python, "-m", "trainspotter.cli", "analyze", path.name,
                   "--fail-on", "error", "--output"]
        proc = subprocess.run(command + ["json"], capture_output=True, text=True,
                              cwd=artifacts, env=env)
        assert proc.returncode in (0, 1, 2), proc.stderr
        report = json.loads(proc.stdout)
        targets = [f for f in report["findings"] if f["detector"] in TARGETS]
        if args.verify:
            metric = report["run"]["metrics"]["train/loss"]
            assert metric.get("unreadable_steps", []) == gaps, name
            assert report["run"]["skipped_values"] == 0
            if gaps:
                assert proc.returncode == 2
                assert report["run"]["skipped_rows"] == len(gaps)
                assert metric["last_step"] == len(patterns[kind]) - 1
                assert all(f["title"] != "Non-finite training loss" for f in targets)
            if kind in ("captured", "rise", "spike", "setback"):
                assert targets == [], (name, targets)
            elif kind == "local":
                assert [(f["title"], f["step_start"], f["step_end"]) for f in targets] == [
                    ("Setback", 100, 120)]
                assert targets[0]["evidence"]["segment_start_step"] == 40
            elif kind == "finite":
                assert [(f["title"], f["step_start"], f["step_end"]) for f in targets] == [
                    ("Setback", 60, 80)]
            elif kind == "nonfinite":
                assert proc.returncode == 1
                assert any(f["title"] == "Non-finite training loss"
                           and f["evidence"]["count"] == 20 for f in targets)
            elif kind == "sparse":
                assert metric["count"] == 2 and metric["last_step"] == 3
                assert report["summary"]["input_complete"]
            elif kind == "explicit":
                assert metric["first_step"] == 100 and metric["last_step"] == 200
                assert report["run"]["skipped_rows"] == 1 and proc.returncode == 2
        html = subprocess.run(command + ["html", "--out", name + ".html"],
                              capture_output=True, text=True, cwd=artifacts, env=env)
        assert html.returncode == proc.returncode, html.stderr
        records.append({"name": name, "inputBytes": len(body), "inputSha256": digest(body),
                        "expectedUnreadableSteps": gaps, "exitCode": proc.returncode,
                        "stderr": proc.stderr, "report": report,
                        "htmlSha256": digest((artifacts / (name + ".html")).read_bytes())})
        print(name, proc.returncode,
              [(f["title"], f["step_start"], f["step_end"]) for f in targets], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runtime": json.loads(runtime.stdout),
                                   "captureInputSha256": digest(original),
                                   "records": records}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
