"""Replay controlled TensorBoard tails through the real CLI and reader.

Requires the tensorboard extra and POSIX resource limits. The 512 MiB address
limit applies only to each child, so a regression cannot exhaust the host.
Usage: python studies/record-lengths/replay.py NEW_DIRECTORY before|after
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

import tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
from tensorboard.summary.writer.record_writer import RecordWriter

CHILD = """
import json,resource,sys
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
if sys.argv[1] == 'probe':
    from trainspotter.readers import load_run
    run = load_run(sys.argv[2], fmt='tensorboard')
    points = run.get('train/loss').points
    print(json.dumps({'readings': [[p.step,p.value,p.wall_time] for p in points],
                      'complete': run.input_complete(), 'unread_bytes': run.unread_event_bytes()}))
else:
    from trainspotter.cli import main
    raise SystemExit(main())
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    phase = sys.argv[2]
    assert phase in {"before", "after"}
    output.mkdir(parents=True)
    fixture = Path(__file__).with_name("prefix.tfevents.fixture")
    prefix = fixture.read_bytes()
    oracle = EventAccumulator(str(fixture), size_guidance={"scalars": 0}).Reload()
    expected = [[e.step, e.value, e.wall_time] for e in oracle.Scalars("loss")]
    assert len(expected) == 20
    buffer = io.BytesIO()
    RecordWriter(buffer).write(Event(
        wall_time=120, step=20,
        summary=Summary(value=[Summary.Value(tag="loss", simple_value=0.25)]),
    ).SerializeToString())
    tail = buffer.getvalue()
    cases = {"complete": prefix}
    for name, length in [("64mib", 2**26), ("memory", 2**60), ("overflow", 2**63),
                         ("uint64-max", 2**64 - 1)]:
        header = struct.pack("<Q", length)
        cases[name] = prefix + header + struct.pack("<I", masked_crc32c(header))
    for name, offset in [("bad-header-crc", 8), ("bad-payload-crc", -1)]:
        damaged = bytearray(tail)
        damaged[offset] ^= 1
        cases[name] = prefix + damaged
    cases["partial-record"] = prefix + tail[:-3]
    cases["short-header"] = prefix + tail[:7]
    results = []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def execute(directory: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-c", CHILD, *arguments], cwd=directory,
                              env=env, text=True, capture_output=True, timeout=30)

    def check(name: str, directory: Path, data: bytes, readings: list[list[float]], complete: bool):
        filename = "events.out.tfevents.fixture"
        (directory / filename).write_bytes(data)
        probe = execute(directory, ["probe", filename])
        cli = execute(directory, ["analyze", filename, "--format", "tensorboard",
                                  "--output", "json", "--fail-on", "error"])
        decoded = json.loads(probe.stdout) if probe.returncode == 0 else None
        report = json.loads(cli.stdout) if cli.stdout.strip() else None
        (directory / "cli.stdout").write_text(cli.stdout)
        (directory / "cli.stderr").write_text(cli.stderr)
        if phase == "before" and name == "memory":
            assert probe.returncode == cli.returncode == 1
            assert decoded is report is None and cli.stderr.rstrip().endswith("MemoryError")
        else:
            assert decoded is not None, probe.stderr
            assert decoded["readings"] == readings
            assert decoded["complete"] is complete
            assert cli.returncode == (0 if complete else 2), cli.stderr
            assert report["run"]["input_complete"] is complete
            assert report["run"]["metrics"]["train/loss"]["count"] == len(readings)
        results.append({
            "case": name, "input_bytes": len(data), "input_sha256": digest(data),
            "probe_exit": probe.returncode, "cli_exit": cli.returncode,
            "report_generated": report is not None,
            "readings": len(decoded["readings"]) if decoded else None,
            "readings_match_accumulator": decoded["readings"] == readings if decoded else None,
            "unread_bytes": decoded["unread_bytes"] if decoded else None,
            "complete": decoded["complete"] if decoded else None,
            "failure": "MemoryError" if cli.stderr.rstrip().endswith("MemoryError") else None,
        })
        assert (directory / filename).read_bytes() == data

    for name, data in cases.items():
        directory = output / name
        directory.mkdir()
        check(name, directory, data, expected, name == "complete")
    recovered = prefix + tail
    # Complete the same previously partial file and load it afresh.
    check("recovered", output / "partial-record", recovered,
          [*expected, [20, 0.25, 120.0]], True)
    directory = output / "memory"
    watch = execute(directory, ["watch", "events.out.tfevents.fixture", "--format", "tensorboard",
                                "--once", "--fail-on", "error", "--no-color"])
    html = execute(directory, ["analyze", "events.out.tfevents.fixture", "--format", "tensorboard",
                               "--output", "html", "--out", "report.html", "--fail-on", "error"])
    (directory / "watch.stdout").write_text(watch.stdout)
    if phase == "after":
        assert watch.returncode == html.returncode == 2
        assert (directory / "report.html").is_file()
    else:
        assert watch.returncode == html.returncode == 1
        assert not (directory / "report.html").exists()
    print(json.dumps({
        "phase": phase, "python": platform.python_version(), "tensorboard": tensorboard.__version__,
        "child_address_limit_mib": 512, "prefix_bytes": len(prefix),
        "prefix_sha256": digest(prefix),
        "oracle": "TensorBoard EventAccumulator, no scalar sampling",
        "cases": results, "watch_exit": watch.returncode, "html_exit": html.returncode,
        "html_generated": (directory / "report.html").is_file(),
        "scope": (
            "Controlled SDK capture with deliberately damaged tails, not a production incident. "
            "Limits apply only to children. No TensorFlow runtime is required."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
