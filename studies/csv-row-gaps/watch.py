"""Verify real watch output as a short CSV record is completed by its writer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    # Reuse the retained ordinary-pipe observer, not a mock of cmd_watch.
    helper = Path(__file__).resolve().parents[1] / "live-stdout/replay.py"
    spec = importlib.util.spec_from_file_location("live_watch_observer", helper)
    assert spec is not None and spec.loader is not None
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    work = args.artifacts.resolve()
    work.mkdir(parents=True, exist_ok=True)
    path = work / "growing.csv"
    values = [1.0] * 50 + [3.0] * 60
    initial = "step,loss,comment\n" + "".join(
        f"{i}\n" if 50 <= i < 60 else f"{i},{v},ok\n" for i, v in enumerate(values)
    )
    completed = "step,loss,comment\n" + "".join(f"{i},{v},ok\n" for i, v in enumerate(values))
    observer.replace(path, initial)
    env = dict(os.environ)
    for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    command = [str(args.python.absolute()), "-m", "trainspotter.cli"]
    watcher = observer.Watch(command, env, work, path.name, 20)
    try:
        assert watcher.wait("unreadable"), watcher.output
        # Let the observed poll and several following polls finish before rewriting.
        time.sleep(0.3)
        assert watcher.wait("Possible plateau"), watcher.output
        before = watcher.output.decode()
        assert "Diverged" not in before, before
        observer.replace(path, completed)
        assert watcher.wait("Read recovered"), watcher.output
        assert watcher.wait("Previously read data changed"), watcher.output
        assert watcher.wait("Diverged (rose from the start)"), watcher.output
    finally:
        result = watcher.finish()
    assert result["exit"] == 0 and result["stderr"] == "", result
    args.out.write_text(
        json.dumps(
            {
                "synthetic": True,
                "unbufferingOverrides": False,
                "initialStdout": before,
                "watch": result,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Live ordinary-pipe watch: gap disclosed, unsupported rise absent, completed rise delivered"
    )


if __name__ == "__main__":
    main()
