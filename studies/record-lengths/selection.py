"""Check that a damaged sibling file does not prevent selecting a healthy run.

Usage: python studies/record-lengths/selection.py NEW_DIRECTORY before|after
Uses the same child memory bound and CLI entrypoint as replay.py.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from replay import CHILD, digest, masked_crc32c


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    phase = sys.argv[2]
    assert phase in {"before", "after"}
    output.mkdir(parents=True)
    prefix = Path(__file__).with_name("prefix.tfevents.fixture").read_bytes()
    header = struct.pack("<Q", 2**60)
    damaged = prefix + header + struct.pack("<I", masked_crc32c(header))
    files = {}
    for name, data in [("healthy", prefix), ("damaged", damaged)]:
        directory = output / name
        directory.mkdir()
        path = directory / "events.out.tfevents.fixture"
        path.write_bytes(data)
        files[path] = data
    cases = []
    for selected in ["healthy", "damaged"]:
        result = subprocess.run(
            [sys.executable, "-c", CHILD, "analyze", ".", "--run", selected,
             "--output", "json", "--fail-on", "error"],
            cwd=output, capture_output=True, text=True, timeout=30,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        report = json.loads(result.stdout) if result.stdout.strip() else None
        if phase == "before":
            assert result.returncode == 1 and report is None
            assert result.stderr.rstrip().endswith("MemoryError")
        else:
            assert result.returncode == (0 if selected == "healthy" else 2), result.stderr
            assert report["run"]["input_complete"] is (selected == "healthy")
            assert report["run"]["metrics"]["train/loss"]["count"] == 20
            damaged_row = next(row for row in report["run"]["event_directories"]
                               if row["directory"] == "damaged")
            assert damaged_row["unread_event_bytes"] == 12
            assert damaged_row["selected"] is (selected == "damaged")
        cases.append({
            "selected": selected, "exit": result.returncode, "report_generated": report is not None,
            "input_complete": report["run"]["input_complete"] if report else None,
            "event_directories": report["run"]["event_directories"] if report else None,
            "failure": "MemoryError" if report is None else None,
        })
    assert all(path.read_bytes() == data for path, data in files.items())
    print(json.dumps({
        "phase": phase, "child_address_limit_mib": 512,
        "files": {str(path.relative_to(output)): {"bytes": len(data), "sha256": digest(data)}
                  for path, data in files.items()},
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
