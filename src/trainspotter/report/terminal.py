"""Plain-ANSI terminal report.

No formatting dependency (numpy is trainspotter's only required package):
this hand-rolls the handful of SGR codes it needs and degrades to plain
text automatically when stdout isn't a TTY or `NO_COLOR`/`--no-color` is
set.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import TextIO

from trainspotter.detectors import Finding
from trainspotter.model import Run

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_ITALIC = "\x1b[3m"
_COLORS = {"red": "\x1b[31m", "yellow": "\x1b[33m", "cyan": "\x1b[36m", "green": "\x1b[32m"}
_SEVERITY_COLOR = {"error": "red", "warning": "yellow", "info": "cyan"}
_SEVERITY_LABEL = {"error": "ERROR", "warning": "WARN ", "info": "INFO "}


def _use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        prefix = "".join(_COLORS.get(c, c) for c in codes)
        return f"{prefix}{text}{_RESET}"


def render_terminal(run: Run, findings: list[Finding], stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    style = _Style(_use_color(stream))
    lines: list[str] = []

    lines.append(style("trainspotter", _BOLD) + style("  run diagnostic", _DIM))
    lines.append(style("-" * 60, _DIM))
    lines.append(f"source   {run.source_path or '(stdin)'}  " + style(f"[{run.source_format}]", _DIM))
    lines.append("metrics  " + (", ".join(run.metric_names()) or "(none found)"))

    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] += 1
    count_bits = [
        style(f"{counts['error']} error", "red") if counts["error"] else style("0 error", _DIM),
        style(f"{counts['warning']} warning", "yellow") if counts["warning"] else style("0 warning", _DIM),
        style(f"{counts['info']} info", "cyan") if counts["info"] else style("0 info", _DIM),
    ]
    lines.append("findings " + "  ".join(count_bits))
    lines.append("")

    if not findings:
        lines.append(style("No pathologies detected.", "green", _BOLD) + " Clean run, or nothing this detector set covers.")
    else:
        for f in findings:
            step_range = str(f.step_start) if f.step_start == f.step_end else f"{f.step_start}-{f.step_end}"
            sev_color = _SEVERITY_COLOR[f.severity]
            label = style(f"[{_SEVERITY_LABEL[f.severity]}]", sev_color, _BOLD)
            header = f"{label} " + style(f"steps {step_range:<13}", _DIM) + style(f"{f.detector:<12}", _DIM) + style(f.title, _BOLD)
            lines.append(header)
            for wrapped in textwrap.wrap(f.message, width=96, initial_indent="  ", subsequent_indent="  "):
                lines.append(wrapped)
            lines.append(style(f"  metric: {f.metric}", _DIM))
            if f.fixes:
                lines.append(style(f"  fix: {f.fixes[0]}", _DIM, _ITALIC))
            lines.append("")

    stream.write("\n".join(lines).rstrip() + "\n")
