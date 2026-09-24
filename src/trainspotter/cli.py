"""Command-line entry point: `trainspotter analyze` and `trainspotter watch`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from trainspotter.detectors import Finding, Severity, run_all, severity_at_least
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report import render_html, render_terminal, to_json_str

_FORMAT_CHOICES = ("hf", "csv", "jsonl", "wandb", "lightning", "tensorboard")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainspotter", description="Diagnose loss curves and training logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a training log and report findings.")
    analyze.add_argument("path", help="Path to a log file (or a TensorBoard logdir).")
    analyze.add_argument("--format", choices=_FORMAT_CHOICES, default=None, help="Force a reader instead of auto-detecting from the path.")
    analyze.add_argument("--output", choices=("terminal", "json", "html"), default="terminal")
    analyze.add_argument("--out", default=None, help="Write the report to this file instead of stdout (required for html unless --output is terminal).")
    analyze.add_argument("--fail-on", choices=("warning", "error"), default=None, help="Exit 1 if any finding at or above this severity was found -- for CI gating.")
    analyze.add_argument("--no-color", action="store_true", help="Disable ANSI color in the terminal report.")
    analyze.add_argument("--title", default=None, help="Title for the HTML report (default: derived from the input path).")

    watch = sub.add_parser("watch", help="Tail a growing log file and print findings as they first appear.")
    watch.add_argument("path", help="Path to a log file being actively written.")
    watch.add_argument("--format", choices=_FORMAT_CHOICES, default=None)
    watch.add_argument("--interval", type=float, default=2.0, help="Seconds between re-reads (default: 2.0).")
    watch.add_argument("--fail-on", choices=("warning", "error"), default=None, help="Stop and exit 1 as soon as a finding at or above this severity appears.")
    watch.add_argument("--once", action="store_true", help="Read and report once instead of looping (used by tests / one-shot checks).")
    watch.add_argument("--no-color", action="store_true")

    return parser


def _emit(run: Run, findings: list[Finding], args: argparse.Namespace) -> None:
    if args.output == "json":
        text = to_json_str(run, findings)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
    elif args.output == "html":
        title = args.title or f"trainspotter &middot; {Path(run.source_path).name or 'report'}"
        text = render_html(run, findings, title=title)
        out_path = Path(args.out) if args.out else Path(run.source_path).with_suffix(".trainspotter.html")
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        import os

        if args.no_color:
            os.environ["NO_COLOR"] = "1"
        render_terminal(run, findings)


def _fail_code(findings: list[Finding], fail_on: Severity | None) -> int:
    if fail_on is None:
        return 0
    return 1 if any(severity_at_least(f.severity, fail_on) for f in findings) else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    run = load_run(args.path, fmt=args.format)
    findings = run_all(run)
    _emit(run, findings, args)
    return _fail_code(findings, args.fail_on)


def cmd_watch(args: argparse.Namespace) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    path = Path(args.path)
    exit_code = 0
    print(f"watching {path} (Ctrl-C to stop)")
    try:
        while True:
            if not path.exists():
                time.sleep(args.interval)
                continue
            try:
                run = load_run(path, fmt=args.format)
            except (ValueError, OSError) as exc:
                # A file being written mid-flush can be transiently unparsable;
                # skip this poll instead of crashing the watcher.
                print(f"  (skipped this read: {exc})")
                if args.once:
                    break
                time.sleep(args.interval)
                continue
            findings = run_all(run)
            new = [f for f in findings if (f.detector, f.metric, f.step_start, f.step_end) not in seen]
            for f in new:
                seen.add((f.detector, f.metric, f.step_start, f.step_end))
                step_range = str(f.step_start) if f.step_start == f.step_end else f"{f.step_start}-{f.step_end}"
                print(f"[{f.severity.upper():7}] steps {step_range:<13} {f.detector:<12} {f.title}: {f.message}")
                if args.fail_on and severity_at_least(f.severity, args.fail_on):
                    exit_code = 1
            if args.once or exit_code:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return cmd_analyze(args)
        if args.command == "watch":
            return cmd_watch(args)
    except (OSError, ValueError) as exc:
        print(f"trainspotter: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
