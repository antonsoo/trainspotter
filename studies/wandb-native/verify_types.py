"""Retain native JSON lexemes and replay the unpublished reader's coercion defect."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
import shutil
import tempfile
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(path: Path) -> dict:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    original = sha(path)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "copy.wandb"
        shutil.copyfile(path, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        while (data := store.scan_data()) is not None:
            record = pb.Record.FromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            items = {
                item.key or ".".join(item.nested_key): item.value_json
                for item in record.history.item
            }
            raw = items["train/loss"]
            value = json.loads(raw)
            rows.append(
                {
                    "step": json.loads(items["global_step"]),
                    "value_json": raw,
                    "json_type": "string" if isinstance(value, str) else "number",
                }
            )
        store.close()
    assert sha(path) == original
    assert [row["json_type"] for row in rows] == ["number", "string", "string", "number"]
    return {
        "input_sha256": original,
        "histories": rows,
        "decoder": "official W&B 0.23.1 DataStore and persistent protobuf",
        "decoder_source_sha256": sha(Path(inspect.getsourcefile(DataStore))),
    }


def verify(path: Path, gold: Path, before: bool, html: Path | None) -> dict:
    import trainspotter
    from trainspotter import load_run
    from trainspotter.cli import main
    from trainspotter.detectors import run_all
    from trainspotter.report import render_html

    expected = json.loads(gold.read_text())
    assert sha(path) == expected["input_sha256"]
    run = load_run(path)
    points = [p.value for p in run.get("train/loss").points]
    divergence = [f for f in run_all(run) if f.detector == "divergence"]
    if before:
        assert points[:2] == [1, 0.5] and math.isnan(points[2]) and points[3] == 0.1
        assert run.input_complete() and divergence
    else:
        assert points == [1, None, None, 0.1]
        assert not run.input_complete() and not divergence and run.skipped_values() == 2
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        gate = main(["analyze", str(path), "--output", "json", "--fail-on", "error"])
    assert gate == (1 if before else 2)
    report = json.loads(output.getvalue())
    assert report["run"]["input_complete"] == run.input_complete()
    if html is not None:
        html.write_text(render_html(run, run_all(run)))
    root = Path(trainspotter.__file__).parent
    return {
        "input_sha256": sha(path),
        "oracle_sha256": sha(gold),
        "scope": "SDK-written type control; before is unpublished ff96e21, not PyPI",
        "values": ["observed NaN" if v is not None and math.isnan(v) else v for v in points],
        "input_complete": run.input_complete(),
        "gate": gate,
        "divergence_findings": len(divergence),
        "runtime_modules": {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["oracle", "verify"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    result = (
        oracle(args.input)
        if args.mode == "oracle"
        else verify(args.input, args.oracle, args.before, args.html)
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.mode, "passed")


if __name__ == "__main__":
    main()
