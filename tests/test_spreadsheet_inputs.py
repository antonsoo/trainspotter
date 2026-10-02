"""Logs as spreadsheets and Windows tools save them: the same run has to give the same report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trainspotter.cli import main
from trainspotter.detectors import run_all
from trainspotter.readers import load_run

EXAMPLES = Path(__file__).parent.parent / "examples"
PLAIN = (EXAMPLES / "divergence.csv").read_text(encoding="utf-8")


def _findings(path: Path) -> list[tuple[str, int, int, str]]:
    run = load_run(path)
    return [(f.detector, f.step_start, f.step_end, f.severity) for f in run_all(run)]


EXPECTED = _findings(EXAMPLES / "divergence.csv")


def _semicolons(text: str) -> str:
    header, *rows = text.splitlines()
    converted = [";".join(cell.replace(".", ",") for cell in row.split(",")) for row in rows]
    return "\n".join([header.replace(",", ";"), *converted]) + "\n"


HEADER_COMMAS = PLAIN.splitlines()[0].count(",")
VARIANTS = {
    # Excel's "CSV UTF-8"
    "byte-order mark and CRLF": b"\xef\xbb\xbf" + PLAIN.replace("\n", "\r\n").encode(),
    "semicolons and decimal commas": _semicolons(PLAIN).encode(),
    "tabs": PLAIN.replace(",", "\t").encode(),
    "a space after each comma of the header": PLAIN.replace(",", ", ", HEADER_COMMAS).encode(),
}


def test_the_example_has_findings_to_compare() -> None:
    assert len(EXPECTED) >= 4
    assert any(detector == "divergence" for detector, *_ in EXPECTED)


@pytest.mark.parametrize("label", list(VARIANTS))
def test_a_csv_saved_another_way_gives_the_same_findings(tmp_path: Path, label: str) -> None:
    # With the mark, "step" stopped being the step column: every finding moved to its row
    # number. With semicolons, no metric was found and the run was reported clean.
    path = tmp_path / "run.csv"
    path.write_bytes(VARIANTS[label])
    assert _findings(path) == EXPECTED
    assert "step" not in load_run(path).metric_names()


def test_json_logs_with_a_byte_order_mark(tmp_path: Path) -> None:
    state = (EXAMPLES / "divergence.trainer_state.json").read_bytes()
    expected = _findings(EXAMPLES / "divergence.trainer_state.json")
    path = tmp_path / "trainer_state.json"
    path.write_bytes(b"\xef\xbb\xbf" + state)
    assert _findings(path) == expected
    rows = [json.dumps(row) for row in json.loads(state)["log_history"]]
    jsonl = tmp_path / "log.jsonl"
    jsonl.write_bytes(b"\xef\xbb\xbf" + "\n".join(rows).encode())
    assert load_run(jsonl).meta["skipped_lines"] == 0


def test_a_log_with_no_metrics_is_refused_not_reported_clean(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "notes.csv"
    path.write_text("date,comment\n2026-10-01,started\n2026-10-02,diverged\n", encoding="utf-8")
    code = main(["analyze", str(path), "--fail-on", "error"])
    captured = capsys.readouterr()
    assert code == 2  # not 0: with --fail-on, a log that can't be read must not pass
    assert captured.out == ""
    assert "no numeric metrics found" in captured.err
    assert "Its columns: date, comment." in captured.err
    assert "No pathologies" not in captured.err


def test_a_pipe_carries_any_metric_name(tmp_path: Path) -> None:
    # A redirected stdout on Windows has the system's code page: a metric named in Chinese
    # ended the report with "'charmap' codec can't encode characters".
    name = "训练/loss"
    path = tmp_path / "run.csv"
    rows = [f"{step},{2.0 * 0.99 ** (step / 10):.5f}" for step in range(0, 600, 10)]
    path.write_text(f"step,{name}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"
    command = [sys.executable, "-m", "trainspotter.cli", "analyze", str(path)]
    result = subprocess.run(command, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert name in result.stdout.decode("utf-8")
