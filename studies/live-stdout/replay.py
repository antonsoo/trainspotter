"""Observe real watch processes through ordinary pipes and redirected files.

Synthetic inputs only. No -u, PYTHONUNBUFFERED, stdbuf, or pseudo-terminal.
Run --probe-only against the baseline; the complete replay requires the repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


def replace(path: Path, contents: str) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def losses(values: list[float], *, partial: bool = False) -> str:
    return "".join(json.dumps({"step": i, "loss": v}) + "\n" for i, v in enumerate(values)) + (
        "{partial\n" if partial else ""
    )


class Watch:
    def __init__(self, command: list[str], env: dict[str, str], cwd: Path, name: str):
        self.command = [*command, "watch", name, "--interval", "0.05"]
        self.started = time.monotonic()
        self.process = subprocess.Popen(
            self.command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.selector = selectors.DefaultSelector()
        assert self.process.stdout is not None
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.output = b""
        self.milestones: list[dict[str, object]] = []

    def wait(self, phrase: str, count: int = 1, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        while self.output.count(phrase.encode("utf-8")) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.process.poll() is not None:
                return False
            for key, _ in self.selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    return False
                self.output += chunk
        alive = self.process.poll() is None
        self.milestones.append({
            "phrase": phrase, "count": count, "alive": alive,
            "seconds_since_start": round(time.monotonic() - self.started, 6),
        })
        return alive

    def finish(self) -> dict[str, object]:
        before_interrupt = self.output.decode("utf-8")
        alive = self.process.poll() is None
        if alive:
            self.process.send_signal(signal.SIGINT)
        try:
            tail, stderr = self.process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            raise
        finally:
            self.selector.close()
        self.output += tail
        return {
            "arguments": self.command[self.command.index("-m") + 2:],
            "alive_before_interrupt": alive,
            "stdout_before_interrupt": before_interrupt,
            "stdout": self.output.decode("utf-8"), "stderr": stderr.decode("utf-8"),
            "exit": self.process.returncode, "milestones": self.milestones,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path,
                        help="Checkout root; omit for isolated installed CLI")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-seconds", type=float, default=10)
    args = parser.parse_args()
    work = args.work_dir.absolute()
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    command = [str(args.python.absolute())]
    if args.source:
        env["PYTHONPATH"] = str(args.source.absolute() / "src")
    else:
        command.append("-I")
    command += ["-m", "trainspotter.cli"]
    metadata = subprocess.check_output(
        [*command[:-2], "-c", "import sys,numpy; print(sys.version); print(numpy.__version__)"],
        env=env, cwd=work, text=True,
    ).splitlines()
    report: dict[str, object] = {
        "platform": platform.platform(), "python": metadata[0], "numpy": metadata[1],
        "mode": "source" if args.source else "isolated installed wheel",
        "synthetic": True, "unbuffering_overrides": False,
    }
    path = work / "train-λ.jsonl"
    values = [2.0, 1.8, float("nan"), 1.6, 1.4]
    replace(path, losses(values))
    watcher = Watch(command, env, work, path.name)
    try:
        visible = watcher.wait("Non-finite training loss", timeout=args.probe_seconds)
    finally:
        probe = watcher.finish()
    probe["finding_visible_while_alive"] = visible
    probe["observation_limit_seconds"] = args.probe_seconds
    assert probe["exit"] == 0 and probe["stderr"] == "", probe
    assert "Non-finite training loss" in str(probe["stdout"]), probe
    report["initial_finding"] = probe

    if not args.probe_only:
        assert visible, probe
        path.unlink()
        watcher = Watch(command, env, work, path.name)
        try:
            assert watcher.wait("skipped this read"), watcher.output
            replace(path, "{partial\n")
            assert watcher.wait("skipped this read", 2), watcher.output
            replace(path, losses(values))
            assert watcher.wait("Read recovered"), watcher.output
            assert watcher.wait("Non-finite training loss"), watcher.output
            values.append(float("nan"))
            replace(path, losses(values, partial=True))
            assert watcher.wait("input    ", 2), watcher.output
            assert watcher.output.count(b"Non-finite training loss") == 1
            replace(path, losses(values))
            assert watcher.wait("Read recovered", 2), watcher.output
            values[0] = 2.1
            replace(path, losses(values))
            assert watcher.wait("Previously read data changed"), watcher.output
            assert watcher.wait("Non-finite training loss", 2), watcher.output
            path.unlink()
            assert watcher.wait("skipped this read", 3), watcher.output
        finally:
            lifecycle = watcher.finish()
        assert lifecycle["exit"] == 2 and lifecycle["stderr"] == "", lifecycle
        assert str(lifecycle["stdout"]).count("Non-finite training loss") == 2, lifecycle
        report["recovery_deduplication_and_rewrite"] = lifecycle

        chart = work / "chart.csv"
        replace(chart, "Step,a - loss,a - loss__MIN,a - loss__MAX\n0,nan,nan,nan\n")
        watcher = Watch(command, env, work, chart.name)
        try:
            assert watcher.wait("a (selection held while watching)"), watcher.output
            assert watcher.wait("Non-finite training loss"), watcher.output
        finally:
            selected = watcher.finish()
        assert selected["exit"] == 0 and selected["stderr"] == "", selected
        report["selected_run"] = selected

        replace(path, losses(values))
        redirected = work / "redirected.txt"
        with redirected.open("wb") as stream:
            process = subprocess.Popen(
                [*command, "watch", path.name, "--interval", "0.05"],
                cwd=work, env=env, stdout=stream, stderr=subprocess.PIPE,
            )
            started = time.monotonic()
            before = b""
            try:
                while time.monotonic() - started < 30 and process.poll() is None:
                    before = redirected.read_bytes()
                    if b"Non-finite training loss" in before:
                        break
                    time.sleep(0.01)
                alive = process.poll() is None
                assert alive and b"Non-finite training loss" in before, before
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    _, stderr = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    raise
        assert process.returncode == 0 and stderr == b"", stderr
        report["redirected_file"] = {
            "alive_when_finding_read": alive, "stdout_before_interrupt": before.decode(),
            "stdout": redirected.read_text(), "exit": process.returncode,
        }

    # Same fixture bytes and relative filenames in both source revisions.
    replace(path, losses([2.0, 1.8, float("nan"), 1.6, 1.4]))
    replace(work / "partial.jsonl", losses([2.0, 1.8], partial=True))
    replace(work / "clean.jsonl", losses([2.0, 1.8, 1.6]))
    cases = [
        (["watch", path.name, "--once"], 0),
        (["watch", path.name, "--once", "--fail-on", "error"], 1),
        (["watch", "missing.jsonl", "--once"], 2),
        (["watch", "partial.jsonl", "--once", "--fail-on", "error"], 2),
        (["watch", "clean.jsonl", "--once", "--require-timing"], 2),
        (["watch", "clean.jsonl", "--once"], 0),
        (["analyze", path.name, "--output", "json"], 0),
        (["analyze", "partial.jsonl", "--output", "json", "--fail-on", "error"], 2),
        (["analyze", "clean.jsonl", "--output", "terminal", "--no-color"], 0),
    ]
    controls = []
    for arguments, expected in cases:
        result = subprocess.run(
            [*command, *arguments], env=env, cwd=work, capture_output=True, timeout=30,
        )
        assert result.returncode == expected, (arguments, result)
        controls.append({
            "arguments": arguments, "exit": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode(),
        })
    report["completed_command_controls"] = controls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"live_finding": visible, "controls": len(controls)}))


if __name__ == "__main__":
    main()
