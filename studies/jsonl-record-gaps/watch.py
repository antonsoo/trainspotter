"""Observe real JSONL watch recovery over an ordinary stdout pipe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).resolve().parents[1] / "live-stdout/replay.py"
    spec = importlib.util.spec_from_file_location("live_watch_observer", helper)
    assert spec is not None and spec.loader is not None
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    work = args.artifacts.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    command = [str(args.python.absolute())]
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    else:
        command.append("-I")
    command += ["-m", "trainspotter.cli"]
    path = work / "growing.jsonl"
    values = [1.] * 50 + [3.] * 60
    initial = "".join('{"loss":\n' if 50 <= i < 60 else json.dumps({"loss": value}) + "\n"
                      for i, value in enumerate(values))
    completed = "".join(json.dumps({"loss": value}) + "\n" for value in values)
    observer.replace(path, initial)
    watcher = observer.Watch(command, env, work, path.name, 20)
    try:
        assert watcher.wait("unreadable"), watcher.output
        assert watcher.wait("Possible plateau"), watcher.output
        time.sleep(0.3)
        before = watcher.output.decode()
        assert "Diverged" not in before, before
        observer.replace(path, completed)
        assert watcher.wait("Previously read data changed"), watcher.output
        assert watcher.wait("Diverged (rose from the start)"), watcher.output
        assert b"input    The horizontal axis uses (row index)" in watcher.output
    finally:
        rewrite = watcher.finish()
    assert rewrite["exit"] == 0 and rewrite["stderr"] == "", rewrite

    # A writer can finish its final JSON object by appending bytes alone. The
    # completed reading must replace its unknown position, without shifting it.
    path = work / "tail.jsonl"
    path.write_text('{"loss":1}\n{"loss":')
    watcher = observer.Watch(command, env, work, path.name, 20)
    try:
        assert watcher.wait("unreadable"), watcher.output
        time.sleep(0.3)
        with path.open("a") as writer:
            writer.write("2}\n")
        assert watcher.wait("Previously read data changed"), watcher.output
        assert b"input    The horizontal axis uses (row index)" in watcher.output
    finally:
        append = watcher.finish()
    assert append["exit"] == 0 and append["stderr"] == "", append
    probe = subprocess.run(command + ["analyze", path.name, "--output", "json",
                                      "--fail-on", "error"],
                           cwd=work, env=env, capture_output=True, text=True, check=True)
    report = json.loads(probe.stdout)
    assert report["summary"]["input_complete"]
    assert report["run"]["metrics"]["train/loss"] == {
        "count": 2, "analysis_count": 2, "distinct_steps": 2,
        "analysis_excluded": False, "first_step": 0, "last_step": 1,
    }
    args.out.write_text(json.dumps({"synthetic": True, "unbufferingOverrides": False,
                                   "initialStdout": before, "rewrite": rewrite,
                                   "append": append, "completedTailReport": report},
                                  indent=2) + "\n")
    print("Ordinary-pipe watch: gap disclosed, completed trajectory delivered; "
          "append-only record completion keeps position 1")


if __name__ == "__main__":
    main()
