"""What the reports say about the log itself: rows that could not be read, the
step range, and that a long or diverged run still gives a usable report."""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import pytest
from conftest import make_run

from trainspotter.cli import main
from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report import render_html, render_terminal, to_json_dict
from trainspotter.report.html_report import _thin_line


def _log_with_two_bad_lines(tmp_path: Path) -> Path:
    path = tmp_path / "log.jsonl"
    lines = [json.dumps({"step": i * 10 + 100, "loss": 2.0 - i * 0.01}) for i in range(40)]
    lines[5] = "{ not json"
    lines[9] = json.dumps({"step": float("nan"), "loss": 1.0})
    path.write_text("\n".join(lines))
    return path


def test_every_report_says_how_many_rows_were_skipped(tmp_path: Path) -> None:
    run = load_run(_log_with_two_bad_lines(tmp_path))
    findings = run_all(run)
    assert run.skipped_rows() == 2

    assert to_json_dict(run, findings)["run"]["skipped_rows"] == 2
    terminal = io.StringIO()
    render_terminal(run, findings, stream=terminal)
    assert "skipped  2 rows could not be read and are not in this report" in terminal.getvalue()
    assert "<dt>skipped</dt>" in render_html(run, findings)
    assert "2 rows could not be read" in render_html(run, findings)


def test_a_log_read_in_full_has_no_skipped_line() -> None:
    run = make_run(**{"train/loss": [2.0, 1.5, 1.2, 1.0]})
    terminal = io.StringIO()
    render_terminal(run, [], stream=terminal)
    assert "skipped" not in terminal.getvalue()
    assert "<dt>skipped</dt>" not in render_html(run, [])
    assert to_json_dict(run, [])["run"]["skipped_rows"] == 0


def test_html_step_range_is_the_range_logged(tmp_path: Path) -> None:
    # It used to read "0-490" for a log whose first step is 100.
    run = load_run(_log_with_two_bad_lines(tmp_path))
    assert "<dt>steps</dt><dd>100&ndash;490</dd>" in render_html(run, [])


def test_a_diverged_run_prints_nothing_on_stderr(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    # numpy used to print "RuntimeWarning: invalid value encountered in subtract" above it.
    rows = ["step,loss,grad_norm,samples_per_second"]
    for i in range(300):
        loss = 2.5 * math.exp(-i / 80) + 0.2 if i < 220 else (math.inf if i < 240 else math.nan)
        rows.append(f"{i * 10},{loss},{1.0 if i < 215 else math.nan},{0 if i % 50 == 0 else 120}")
    path = tmp_path / "diverged.csv"
    path.write_text("\n".join(rows) + "\n")
    assert main(["analyze", str(path), "--no-color"]) == 0
    captured = capfd.readouterr()
    assert "Diverged (non-finite values)" in captured.out
    assert captured.err == ""


def test_thin_line_keeps_the_points_that_shape_each_column() -> None:
    sparse = [(float(x), float(x % 7)) for x in range(0, 900, 3)]
    assert _thin_line(sparse) == sparse

    # 1,000 points in one half-pixel column: first, lowest, highest, last survive, in order.
    column = [(10.0 + i * 0.0001, 50.0 + (i * 37) % 11) for i in range(1000)]
    column[400] = (column[400][0], 3.0)
    column[700] = (column[700][0], 140.0)
    assert _thin_line(column) == [column[0], column[400], column[700], column[-1]]


def test_a_long_run_gives_a_report_a_browser_can_open() -> None:
    n = 200_000
    values = [2.5 * math.exp(-i / 50_000) + 0.3 + 0.02 * math.sin(i) for i in range(n)]
    values[123_456] = 4.0  # one spike, a single step wide
    run = make_run(**{"train/loss": values})
    html = render_html(run, [])
    assert len(html) < 150_000, f"{len(html)} bytes"  # with every point in the path: 2.6 MB
    drawn = re.findall(r'<path d="([^"]*)" fill="none"', html)
    assert len(drawn) == 1
    ys = [float(pair.split()[1]) for pair in drawn[0][1:].split(" L")]
    # The spike is the highest point of the run, so it is the smallest y in the path.
    x_of_top = float(drawn[0][1:].split(" L")[ys.index(min(ys))].split()[0])
    assert x_of_top == pytest.approx(54 + 123_456 / (n - 1) * (920 - 54 - 18), abs=0.11)
