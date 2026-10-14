"""Replay native writer counters against creation-order TensorBoard accumulation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.summary.writer.record_writer import RecordWriter

STUDY = Path(__file__).resolve().parent


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, list | tuple):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path)
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
    from trainspotter.readers import load_run
    from trainspotter.report import to_json_dict

    origin = Path(trainspotter.__file__).resolve()
    assert (
        origin.is_relative_to(args.src.resolve()) if args.src else "site-packages" in origin.parts
    )
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    logging.getLogger("tensorboard").setLevel(logging.ERROR)
    controlled = json.loads((STUDY / "fixtures/controlled-capture.json").read_text())["captures"]
    native_manifest = json.loads((STUDY / "fixtures/native-capture.json").read_text())
    records, cases = [], []

    def add(label, files, order, *, complete=True, fallback=False):
        cases.append(
            {
                "id": label,
                "files": files,
                "order": sorted(order) if fallback else order,
                "complete": complete,
                "scope": "Constructed file-layout/damage variant of native SDK records",
            }
        )

    for label, manifest in controlled.items():
        files = {
            name: (STUDY / "fixtures" / label / name).read_bytes()
            for name in manifest["created_order"]
        }
        for name, data in files.items():
            assert hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"]
        order = manifest["created_order"]
        add(label, files, order)
        add(label + "-under-ten", {name: files[name] for name in order[:9]}, order[:9])
        add(label + "-single-file", {order[-1]: files[order[-1]]}, order[-1:])
        for suffix in [".native", "_capture", ".native.123.456"]:
            renamed = {name + suffix: data for name, data in files.items()}
            add(label + "-suffix-" + suffix, renamed, [name + suffix for name in order])
        named = {
            f"events.out.tfevents.1790000000.writer.1.2.domain.999.{i}.native.123.456": files[name]
            for i, name in enumerate(order)
        }
        add(label + "-numeric-host", named, list(named))
        timed = {
            f"events.out.tfevents.{1790000000 + i}.writer.999.{i}": files[name]
            for i, name in enumerate(order)
        }
        add(label + "-different-seconds", timed, list(timed))
        manual = {
            f"events.out.tfevents.{i:010d}.manual": files[name] for i, name in enumerate(order)
        }
        add(label + "-manual-names", manual, list(manual))
        damaged = dict(files)
        value = bytearray(damaged[order[10]])
        value[8] ^= 1
        damaged[order[10]] = bytes(value)
        add(label + "-header-crc", damaged, order, complete=False, fallback=True)
        short = dict(files)
        short[order[10]] = short[order[10]][:10]
        add(label + "-partial-header", short, order, complete=False, fallback=True)
        split = {"train/" + name: data for name, data in files.items()}
        add(label + "-split-directory", split, ["train/" + name for name in order])
    files = {
        name: (STUDY / "fixtures/native" / name).read_bytes()
        for name in native_manifest["createdOrder"]
    }
    add("unmodified-native-capture", files, native_manifest["createdOrder"])
    cases[-1]["scope"] = (
        "Unmodified SDK capture with an uncontrolled real clock and no model training"
    )

    for item in cases:
        folder = args.out / item["id"]
        source = folder / "input"
        source.mkdir(parents=True)
        for name, data in item["files"].items():
            path = source / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
        # Independent SDK CRC validation per file, then the SDK accumulator over
        # known creation order. This bypasses its directory's own lexical bug.
        buffer = io.BytesIO()
        writer = RecordWriter(buffer)
        for name in item["order"]:
            for record in RawEventFileLoader(str(source / name)).Load():
                writer.write(record)
        oracle_file = folder / "oracle.tfevents.fixture"
        oracle_file.write_bytes(buffer.getvalue())
        oracle = EventAccumulator(str(oracle_file), size_guidance={"scalars": 0}).Reload()
        expected = safe(
            [[point.step, point.value, point.wall_time] for point in oracle.Scalars("loss")]
        )
        has_nonfinite = any(point[1] in ("NaN", "Infinity", "-Infinity") for point in expected)
        gate = 2 if not item["complete"] else 1 if has_nonfinite else 0
        previous = Path.cwd()
        os.chdir(folder)
        run = load_run("input", fmt="tensorboard")
        findings = run_all(run)
        report = to_json_dict(run, findings)
        actual = safe(
            [[point.step, point.value, point.wall_time] for point in run.get("train/loss").points]
        )
        agreement = actual == expected and run.input_complete() == item["complete"]
        records.append(
            {
                "id": item["id"],
                "kind": "library",
                "scope": item["scope"],
                "expected": expected,
                "actual": actual,
                "expectedGate": gate,
                "report": report,
                "agrees": agreement,
            }
        )
        for kind, command in [
            ("cli-json", ["analyze", "input", "--output", "json", "--fail-on", "error"]),
            ("cli-terminal", ["analyze", "input", "--no-color", "--fail-on", "error"]),
            ("watch-once", ["watch", "input", "--once", "--no-color", "--fail-on", "error"]),
            (
                "cli-html",
                [
                    "analyze",
                    "input",
                    "--output",
                    "html",
                    "--out",
                    "report.html",
                    "--fail-on",
                    "error",
                ],
            ),
        ]:
            result = subprocess.run(
                [sys.executable, "-m", "trainspotter.cli", *command, "--format", "tensorboard"],
                env=environment,
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=30,
            )
            body = (
                json.loads(result.stdout)
                if kind == "cli-json" and result.stdout.lstrip().startswith("{")
                else None
            )
            html = (folder / "report.html").read_text() if kind == "cli-html" else None
            agrees = result.returncode == gate
            if kind == "cli-json":
                agrees = agrees and body == report and agreement
            elif kind == "cli-html":
                agrees = agrees and ("Non-finite training loss" in html) == has_nonfinite
            else:
                agrees = agrees and ("Non-finite training loss" in result.stdout) == has_nonfinite
            records.append(
                {
                    "id": item["id"],
                    "kind": kind,
                    "scope": item["scope"],
                    "expectedGate": gate,
                    "command": ["trainspotter", *command, "--format", "tensorboard"],
                    "exit": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "htmlBytes": len(html.encode()) if html is not None else None,
                    "htmlSha256": hashlib.sha256(html.encode()).hexdigest()
                    if html is not None
                    else None,
                    "agrees": agrees,
                }
            )
        os.chdir(previous)
        for name, data in item["files"].items():
            assert (source / name).read_bytes() == data
    result = {
        "python": sys.version.split()[0],
        "tensorboard": importlib.metadata.version("tensorboard"),
        "numpy": importlib.metadata.version("numpy"),
        "originKind": "source" if args.src else "installed",
        "cases": len(cases),
        "workflows": len(records),
        "disagreements": sum(not r["agrees"] for r in records),
        "records": records,
    }
    (args.out / "observations.json").write_text(
        json.dumps(safe(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: result[key] for key in ["cases", "workflows", "disagreements"]}))
    assert args.observe_only or result["disagreements"] == 0


if __name__ == "__main__":
    main()
