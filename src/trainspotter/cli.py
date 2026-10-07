"""Command-line entry point: `trainspotter analyze` and `trainspotter watch`."""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from trainspotter import __version__
from trainspotter.detectors import Finding, Severity, run_all, severity_at_least
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report import render_html, render_terminal, to_json_str
from trainspotter.report.terminal import visible

_FORMAT_CHOICES = ("hf", "csv", "jsonl", "wandb", "lightning", "tensorboard")


def _interval(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 86_400:
        raise argparse.ArgumentTypeError("interval must be greater than zero and at most 86400 seconds")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainspotter", description="Diagnose loss curves and training logs.")
    parser.add_argument("--version", action="version", version=f"trainspotter {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a training log and report findings.")
    analyze.add_argument("path", help="Path to a log file (or a TensorBoard logdir).")
    analyze.add_argument("--format", choices=_FORMAT_CHOICES, default=None, help="Force a reader instead of auto-detecting from the path.")
    analyze.add_argument("--run", default=None, metavar="NAME", help="Which run to analyse when the source holds several (a W&B chart export, a TensorBoard logdir): its name, or the start of it. Default: the run with the most points.")
    analyze.add_argument("--output", choices=("terminal", "json", "html"), default="terminal")
    analyze.add_argument("--out", default=None, help="Write the report to this file (HTML defaults to a .trainspotter.html file beside the source).")
    analyze.add_argument("--fail-on", choices=("warning", "error"), default=None, help="CI gate: exit 1 for findings at this severity, or 2 when input is incomplete.")
    analyze.add_argument("--no-color", action="store_true", help="Disable ANSI color in the terminal report.")
    analyze.add_argument("--title", default=None, help="Title for the HTML report (default: derived from the input path).")

    watch = sub.add_parser("watch", help="Tail a growing log file and print findings as they first appear.")
    watch.add_argument("path", help="Path to a log file being actively written.")
    watch.add_argument("--format", choices=_FORMAT_CHOICES, default=None)
    watch.add_argument("--run", default=None, metavar="NAME", help="Which run to watch when the source holds several.")
    watch.add_argument("--interval", type=_interval, default=2.0, help="Positive seconds between re-reads (default: 2.0).")
    watch.add_argument("--fail-on", choices=("warning", "error"), default=None, help="Stop with exit 1 on a matching finding, or 2 when a loaded run has incomplete input.")
    watch.add_argument("--once", action="store_true", help="Read and report once instead of looping (used by tests / one-shot checks).")
    watch.add_argument("--no-color", action="store_true")

    return parser


def _write_report(path: Path, text: str, source: str) -> None:
    source_path = Path(source)
    if path.resolve() == source_path.resolve() or (
        path.exists() and source_path.exists() and path.samefile(source_path)
    ):
        raise ValueError("report destination is the input file; choose a different --out path")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".trainspotter-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _emit(run: Run, findings: list[Finding], args: argparse.Namespace) -> None:
    if args.output == "json":
        text = to_json_str(run, findings)
        if args.out:
            _write_report(Path(args.out), text + "\n", run.source_path)
        else:
            sys.stdout.write(text + "\n")
    elif args.output == "html":
        # A literal middot, not the &middot; entity: render_html HTML-escapes
        # this title for the <title> tag, which would turn an entity
        # reference into visible text ("&amp;middot;") instead of "·".
        title = args.title or f"trainspotter · {Path(run.source_path).name or 'report'}"
        text = render_html(run, findings, title=title)
        out_path = Path(args.out) if args.out else Path(run.source_path).with_suffix(".trainspotter.html")
        _write_report(out_path, text, run.source_path)
        print(f"wrote {visible(str(out_path))}")
    else:
        if args.out:
            stream = io.StringIO()
            render_terminal(run, findings, stream=stream, color=False)
            _write_report(Path(args.out), stream.getvalue(), run.source_path)
        else:
            render_terminal(run, findings, color=False if args.no_color else None)


def _fail_code(findings: list[Finding], fail_on: Severity | None) -> int:
    if fail_on is None:
        return 0
    return 1 if any(severity_at_least(f.severity, fail_on) for f in findings) else 0


def _require_metrics(run: Run) -> None:
    if not run.metric_names():
        # Not "no pathologies detected": nothing was looked at. With --fail-on in CI, a log
        # that could not be read would otherwise pass.
        columns = run.meta.get("columns")
        seen = (
            f" Its columns: {', '.join(map(str, columns[:12]))}."
            if isinstance(columns, list) and columns
            else ""
        )
        raise ValueError(
            f"no numeric metrics found in {run.source_path} (read as {run.source_format}).{seen} "
            "If that is the wrong format, name it with --format."
        )


def _gate(run: Run, findings: list[Finding], fail_on: Severity | None) -> int:
    if fail_on is not None and not run.input_complete():
        print("trainspotter: input is incomplete; CI gate could not be evaluated", file=sys.stderr)
        return 2
    return _fail_code(findings, fail_on)


def cmd_analyze(args: argparse.Namespace) -> int:
    run = load_run(args.path, fmt=args.format, run_name=args.run)
    _require_metrics(run)
    findings = run_all(run)
    _emit(run, findings, args)
    return _gate(run, findings, args.fail_on)


def _extends(previous: Run, current: Run) -> bool:
    """A rewrite/restart invalidates finding suppression. NaN equals NaN for this comparison."""
    for name, old in previous.metrics.items():
        new = current.get(name)
        if new is None or len(new) < len(old):
            return False
        for before, after in zip(old.points, new.points, strict=False):
            if (before.step, float(before.value).hex(), str(before.wall_time)) != (
                after.step, float(after.value).hex(), str(after.wall_time),
            ):
                return False
    return True


def cmd_watch(args: argparse.Namespace) -> int:
    seen: dict[tuple[str, str, str, int], Severity] = {}
    path = Path(args.path)
    exit_code = 2
    previous: Run | None = None
    selected = args.run
    last_notice: str | None = None
    print(f"watching {visible(str(path))} (Ctrl-C to stop)")
    try:
        while True:
            try:
                run = load_run(path, fmt=args.format, run_name=selected)
                _require_metrics(run)
            except (ValueError, OSError, csv.Error) as exc:
                # A file being written mid-flush can be transiently unparsable;
                # skip this poll instead of crashing the watcher.
                notice = f"  (skipped this read: {visible(str(exc))})"
                if notice != last_notice:
                    print(notice)
                last_notice = notice
                exit_code = 2
                if args.once:
                    return 2
                time.sleep(args.interval)
                continue
            chosen = run.meta.get("selected_run")
            if selected != chosen and isinstance(chosen, str):
                selected = chosen
                print(f"run      {visible(chosen) or '(top level)'} (selection held while watching)")
            notice = " ".join(run.input_notes())
            if notice and notice != last_notice:
                print(f"input    {visible(notice)}")
            elif last_notice and not notice:
                print("input    Read recovered; all supported input was read.")
            last_notice = notice
            if previous is not None and not _extends(previous, run):
                seen.clear()
                print("history  Previously read data changed; findings are evaluated again.")
            findings = run_all(run)
            def identity(f: Finding) -> tuple[str, str, str, int]:
                return f.detector, f.metric, f.title, f.step_start

            new = [f for f in findings if seen.get(identity(f)) != f.severity]
            for f in new:
                step_range = str(f.step_start) if f.step_start == f.step_end else f"{f.step_start}-{f.step_end}"
                print(f"[{f.severity.upper():7}] steps {step_range:<13} {f.detector:<12} {visible(f.title)}: {visible(f.message)}")
            seen = {identity(f): f.severity for f in findings}
            previous = run
            exit_code = _gate(run, findings, args.fail_on)
            if args.once or exit_code:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return exit_code


def _text_streams() -> None:
    """Make stdout and stderr able to carry any text.

    A pipe or a file gets UTF-8: before 3.15, Python on Windows gives it the system's code
    page, where a metric or run name outside it raised ``UnicodeEncodeError``. A terminal keeps its own encoding
    and shows a character it cannot encode as an escape.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # replaced by something that isn't a text file
            continue
        if stream.isatty():
            reconfigure(errors="backslashreplace")
        else:
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _text_streams()
    try:
        if args.command == "analyze":
            return cmd_analyze(args)
        if args.command == "watch":
            return cmd_watch(args)
    except (OSError, ValueError, ImportError, csv.Error) as exc:
        print(f"trainspotter: {visible(str(exc))}", file=sys.stderr)
        return 2
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
