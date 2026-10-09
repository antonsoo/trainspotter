from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from trainspotter.cli import main
from trainspotter.detectors import run_all
from trainspotter.detectors.throughput import ThroughputDetector
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report import render_html, render_terminal, to_json_str
from trainspotter.timing import assess_timing

CAPTURE = Path(__file__).resolve().parents[1] / "examples" / "timing"


def test_cadence_is_not_training_speed_without_an_explicit_step_axis(tmp_path: Path) -> None:
    # The same 0.1-second training steps, observed every 10 then every 50 steps.
    steps = [i * 10 for i in range(21)] + [200 + i * 50 for i in range(1, 21)]
    path = tmp_path / "run.csv"
    path.write_text("wall_time,loss\n" + "".join(f"{1000 + s / 10},{2 - s / 2000}\n"
                                               for s in steps))
    run = load_run(path)
    assert ThroughputDetector().run(run) == []
    evidence = assess_timing(run)
    assert evidence.status == "unavailable"
    assert evidence.axis == "row_index"
    assert evidence.ratio is None
    assert "row indices" in (evidence.reason or "")
    # Restoring the recorded steps gives the unchanged physical rate, not a 5x regression.
    path.write_text("step,wall_time,loss\n" + "".join(f"{s},{1000 + s / 10},{2 - s / 2000}\n"
                                                    for s in steps))
    explicit = assess_timing(load_run(path))
    assert explicit.ratio == pytest.approx(1)
    assert explicit.baseline is not None
    assert explicit.baseline.interval_start_step == 0
    assert explicit.baseline.last_position == 80


@pytest.mark.parametrize("filename,source,unit", [
    ("slow-direct.csv", "step_time", "s/step"),
    ("slow-throughput.csv", "1/throughput", "s/unit"),
])
def test_directly_logged_units_remain_usable_without_steps(
    filename: str, source: str, unit: str,
) -> None:
    run = load_run(CAPTURE / filename)
    timing = assess_timing(run)
    assert timing.axis == "row_index"
    assert timing.status == "compared"
    assert timing.source == source and timing.unit == unit
    assert timing.ratio is not None and timing.ratio > 1.4
    finding, = ThroughputDetector().run(run)
    assert finding.evidence["ratio"] == timing.ratio
    if source == "1/throughput":
        assert timing.source_metric == "samples_per_second"


def test_captured_missing_and_broken_timestamps_have_explicit_unavailable_evidence() -> None:
    missing = assess_timing(load_run(CAPTURE / "slow-missing-time.csv"))
    assert missing.status == "unavailable"
    assert any("1 of 71" in note and "no timestamp" in note for note in missing.notes)
    broken = assess_timing(load_run(CAPTURE / "slow-broken-clock.csv"))
    assert broken.source == "wall_time(train/loss)"
    assert broken.status == "unavailable"
    assert broken.invalid_observations == 1
    assert broken.baseline is not None and broken.recent is not None
    assert broken.baseline.median_seconds is None
    assert broken.baseline.invalid_observations == 1
    assert broken.recent.median_seconds is not None
    assert "not moved" in (broken.reason or "")


def test_bad_preferred_measurements_do_not_fall_back_to_a_convenient_clock() -> None:
    run = Run()
    for step in range(40):
        run.add_point("step_time", step, 0 if step == 2 else 0.1)
        run.add_point("train/loss", step, 1, wall_time=step * 0.1)
    timing = assess_timing(run)
    assert timing.source == "step_time"
    assert timing.status == "unavailable"
    assert timing.ratio is None


def test_invalid_interior_is_retained_without_moving_the_endpoint_windows() -> None:
    run = Run()
    for step in range(40):
        run.add_point("step_time", step, float("nan") if step == 20 else .1)
    timing = assess_timing(run)
    assert timing.status == "compared"
    assert timing.ratio == pytest.approx(1)
    assert timing.invalid_observations == 1
    assert any("outside the two" in note for note in timing.notes)
    assert timing.baseline is not None and timing.recent is not None
    assert (timing.baseline.first_position, timing.baseline.last_position) == (0, 7)
    assert (timing.recent.first_position, timing.recent.last_position) == (32, 39)


def test_reports_expose_timing_even_when_it_has_no_finding() -> None:
    run = load_run(CAPTURE / "stable-without-steps.csv")
    findings = run_all(run)
    data = json.loads(to_json_str(run, findings))
    assert data["timing"]["status"] == "unavailable"
    assert data["timing"]["ratio"] is None
    # Readability and applicability are distinct; missing timing is not corrupt loss data.
    assert data["run"]["input_complete"] is True
    terminal = StringIO()
    render_terminal(run, findings, terminal, color=False)
    assert "row indices" in terminal.getvalue()
    html = render_html(run, findings)
    assert "Timing unavailable" in html and "row indices" in html
    timed = load_run(CAPTURE / "slow-with-steps.csv")
    assert "Timing comparison" in render_html(timed, run_all(timed))
    assert 'aria-label="Timing windows"' in render_html(timed, run_all(timed))


@pytest.mark.parametrize("name,code", [
    ("stable-with-steps", 0), ("slow-with-steps", 1),
    ("stable-without-steps", 2), ("slow-missing-time", 2), ("slow-broken-clock", 2),
])
@pytest.mark.parametrize("command", ["analyze", "watch"])
def test_explicit_timing_gate_distinguishes_a_slowdown_from_unavailable_evidence(
    name: str, code: int, command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "report.json"
    args = [command, str(CAPTURE / f"{name}.csv"), "--fail-on", "warning", "--require-timing"]
    args += ["--once"] if command == "watch" else ["--output", "json", "--out", str(out)]
    assert main(args) == code
    captured = capsys.readouterr()
    if code == 2:
        assert "timing comparison is required" in captured.err
    if command == "analyze":
        data = json.loads(out.read_text())
        assert data["timing"]["status"] == ("unavailable" if code == 2 else "compared")


def test_timing_requirement_alone_checks_availability_and_input_integrity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(CAPTURE / "slow-with-steps.csv"), "--require-timing"]) == 0
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(json.dumps({"step": i, "step_time": .1}) for i in range(20))
                       + "\n{broken\n")
    assert main(["analyze", str(partial), "--require-timing"]) == 2
    assert "input is incomplete" in capsys.readouterr().err


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1e-320])
def test_unusable_rates_keep_standard_json_and_no_invented_ratio(value: float) -> None:
    run = Run()
    for step in range(12):
        run.add_point("throughput", step, value if step < 6 else 1)
    timing = assess_timing(run)
    assert timing.ratio is None
    assert timing.status == "unavailable"
    json.dumps(timing.to_dict(), allow_nan=False)
