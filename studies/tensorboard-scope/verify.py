"""Controlled TensorBoard writer captures, replayed through the library and CLI.

Run with the Python environment containing the package to verify. No trainer,
model download or provider call is involved. --check enforces the corrected behavior.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from trainspotter.readers import load_run


def write(directory: Path, count: int, tag: str = "loss") -> Path:
    writer = EventFileWriter(str(directory))
    for step in range(count):
        writer.add_event(Event(
            step=step, wall_time=100 + step,
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=1)]),
        ))
    writer.close()
    source = next(directory.glob("*tfevents*"))
    destination = directory / "events.out.tfevents.0000000000.capture"
    source.rename(destination)
    return destination


def probe(root: Path, selection: str | None) -> dict[str, object]:
    arguments = ["analyze", ".", "--output", "json", "--fail-on", "error"]
    if selection is not None:
        arguments += ["--run", selection]
    completed = subprocess.run(
        [sys.executable, "-c", "from trainspotter.cli import main; raise SystemExit(main())",
         *arguments], cwd=root, capture_output=True, text=True, timeout=20,
    )
    run = load_run(root, run_name=selection)
    return {
        "complete": run.input_complete(), "unread_bytes": run.unread_event_bytes(),
        "skipped_events": run.skipped_rows(),
        "points": sum(len(series) for series in run.metrics.values()),
        "gate_exit": completed.returncode,
        "traceback": "Traceback" in completed.stderr,
    }


def measure(root: Path) -> dict[str, object]:
    write(root / "selected", 50)
    other = write(root / "unrelated", 50)
    other.write_bytes(other.read_bytes()[:-3])
    results: dict[str, object] = {
        "direct": probe(root / "selected", None),
        "selected_from_parent": probe(root, "selected"),
        "damaged_sibling": probe(root, "unrelated"),
    }
    split = write(root / "selected" / "validation", 2, "loss")
    split.write_bytes(split.read_bytes()[:-3])
    results["damaged_validation"] = probe(root, "selected")
    header = struct.pack("<Q", 2**63)
    overflow = write(root / "overflow", 50)
    with overflow.open("ab") as handle:
        handle.write(header + struct.pack("<I", masked_crc32c(header)))
    try:
        results["overflow_length"] = probe(root / "overflow", None)
    except OverflowError as exc:
        results["overflow_length"] = {"exception": type(exc).__name__, "message": str(exc)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="trainspotter-scope-") as temporary:
        measured = measure(Path(temporary))
    if args.check:
        assert measured["direct"] == measured["selected_from_parent"]
        assert measured["direct"] == {
            "complete": True, "unread_bytes": 0, "skipped_events": 0,
            "points": 50, "gate_exit": 0, "traceback": False,
        }
        for name, unread, points in (
            ("damaged_sibling", 39, 49), ("damaged_validation", 39, 51),
            ("overflow_length", 12, 50),
        ):
            assert measured[name] == {
                "complete": False, "unread_bytes": unread, "skipped_events": 0,
                "points": points, "gate_exit": 2, "traceback": False,
            }
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
