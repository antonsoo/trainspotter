"""Replay malformed CSV rows through the actual JSON and standalone HTML CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMATS = {
    "csv": ["step", "loss", "comment"],
    "lightning": ["step", "train_loss_step", "comment"],
    "wandb": ["_step", "trainer/global_step", "loss", "comment"],
    "chart": ["Step", "run - loss", "run - loss__MIN", "run - loss__MAX"],
    "row-index": ["loss", "comment"],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(values: list[float | None], fmt: str, shape: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(FORMATS[fmt])
    for step, value in enumerate(values):
        row: list[object] = [] if fmt == "row-index" else [step]
        if fmt == "wandb":
            row.append(step)
        if value is None and shape == "short":
            # A physical blank line is ignored by DictReader, so retain a field.
            writer.writerow(row or ["unreadable"])
            continue
        row.append(value if value is not None else "unreadable")
        row.extend(["", ""] if fmt == "chart" else ["note"])
        if value is None and shape == "extra":
            row.append("extra")
        writer.writerow(row)
    return output.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path, help="Source checkout; omit for an installed wheel")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    python = str(args.python.absolute())  # Do not resolve a venv interpreter symlink.
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    probe = subprocess.run(
        [
            python,
            "-c",
            """
import hashlib,json,platform,numpy
from pathlib import Path
import trainspotter
root=Path(trainspotter.__file__).parent
print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,
 'sourceHashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(root.rglob('*.py'))}}))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=artifacts,
    )
    runtime = json.loads(probe.stdout)
    capture = ROOT.parent / "finite-segments/captured"
    source = (capture / "observed.csv").read_bytes()
    manifest = json.loads((capture / "capture.json").read_text())
    assert digest(source) == manifest["files"]["observed.csv"]
    native_gaps = [
        int(row[0]) for row in csv.reader(source.decode().splitlines()[1:]) if row[1] == "nan"
    ]
    assert len(native_gaps) == 20
    cases: list[tuple[str, bytes, str, list[int], str]] = []
    for shape in ("short", "extra"):
        rows = []
        for row in source.decode().splitlines():
            if row.endswith(",nan"):
                row = row.split(",")[0] if shape == "short" else row + ",extra"
            rows.append(row)
        cases.append(
            (f"captured-{shape}", ("\n".join(rows) + "\n").encode(), "csv", native_gaps, "gap")
        )
    patterns: dict[str, list[float | None]] = {
        "rise": [1.0] * 50 + [None] * 10 + [3.0] * 50,
        "spike": [5.0] * 20 + [1.0] * 40 + [None] * 10 + [4.0] + [1.0] * 40,
        "setback": [5.0] * 20 + [1.0] * 40 + [None] * 10 + [4.0] * 20 + [None] * 10 + [1.0] * 40,
        "local": [1.0] * 30 + [None] * 10 + [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40,
    }
    for fmt in FORMATS:
        for shape in ("short", "extra"):
            for kind, values in patterns.items():
                cases.append(
                    (
                        f"{fmt}-{shape}-{kind}",
                        encode(values, fmt, shape),
                        fmt,
                        [i for i, v in enumerate(values) if v is None],
                        kind,
                    )
                )
    cases.append(("observed-nan-control", source, "csv", [], "nonfinite"))
    finite = [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40
    cases.append(("finite-control", encode(finite, "csv", "complete"), "csv", [], "finite"))
    records = []
    for name, body, fmt, gaps, kind in cases:
        path = artifacts / (name + ".csv")
        path.write_bytes(body)
        command = [python, "-m", "trainspotter.cli", "analyze", path.name]
        if fmt == "lightning":
            command.extend(["--format", "lightning"])
        command.extend(["--fail-on", "error", "--output"])
        proc = subprocess.run(
            command + ["json"], capture_output=True, text=True, cwd=artifacts, env=env
        )
        assert proc.returncode in (0, 1, 2), proc.stderr
        report = json.loads(proc.stdout)
        assert report["run"]["source_path"] == path.name
        targets = [
            f for f in report["findings"] if f["detector"] in ("divergence", "spikes", "setbacks")
        ]
        if args.verify:
            assert report["run"]["metrics"]["train/loss"].get("unreadable_steps", []) == gaps
            assert report["run"]["skipped_rows"] == len(gaps)
            assert report["run"]["skipped_values"] == 0
            assert report["summary"]["input_complete"] == (not gaps)
            if gaps:
                assert proc.returncode == 2
                assert all("non-finite" not in f["title"].lower() for f in targets)
            if kind in ("gap", "rise", "spike", "setback"):
                assert targets == [], (name, targets)
            elif kind == "local":
                assert [(f["title"], f["step_start"], f["step_end"]) for f in targets] == [
                    ("Setback", 100, 120)
                ]
                assert targets[0]["evidence"]["segment_start_step"] == 40
            elif kind == "nonfinite":
                assert proc.returncode == 1
                assert any(
                    f["title"] == "Non-finite training loss" and f["evidence"]["count"] == 20
                    for f in targets
                )
            elif kind == "finite":
                assert any(
                    f["title"] == "Setback" and f["step_start"] == 60 and f["step_end"] == 80
                    for f in targets
                )
        html_path = artifacts / (name + ".html")
        html = subprocess.run(
            command + ["html", "--out", html_path.name],
            capture_output=True,
            text=True,
            cwd=artifacts,
            env=env,
        )
        assert html.returncode == proc.returncode, html.stderr
        records.append(
            {
                "name": name,
                "inputBytes": len(body),
                "inputSha256": digest(body),
                "expectedUnreadableSteps": gaps,
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
        "captureFault": "At exactly the 20 captured nan rows, remove the loss cell "
        "or append one extra field. Other rows remain byte-identical.",
        "records": records,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
