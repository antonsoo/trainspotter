"""Logging more or less often must not change inferred training speed."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from trainspotter.cli import main
from trainspotter.detectors.throughput import ThroughputDetector
from trainspotter.readers import load_run
from trainspotter.report import render_html, render_terminal, to_json_str
from trainspotter.timing import assess_timing


def history(path: Path, *, counter: str | None = "trainer/global_step", slow: bool = False) -> None:
    """A numerical control: 0.1 seconds/update, logged every 10 then every 50 updates."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["_step", "_runtime", "train/loss", *([counter] if counter else [])])
        for position in range(61):
            step = position * 10 if position <= 30 else 300 + (position - 30) * 50
            elapsed = step / 10 + (max(0, step - 300) / 5 if slow else 0)
            writer.writerow([position, elapsed, 1 + 1 / (step + 10),
                             *([step] if counter else [])])


@pytest.mark.parametrize("counter", [
    "trainer/global_step", "train/global_step", "global_step", "step",
])
def test_native_training_axis_removes_false_cadence_alarm(tmp_path: Path, counter: str) -> None:
    path = tmp_path / "history.csv"
    history(path, counter=counter)
    run = load_run(path)
    assert run.meta["step_column"] == counter
    assert run.axis_kind() == "step"
    assert run.metric_names() == ["train/loss"]
    assert assess_timing(run).ratio == pytest.approx(1)
    assert ThroughputDetector().run(run) == []
    series = run.get("train/loss")
    assert series is not None and series.steps()[-1] == 1800
    # The identical logger cadence must still expose a genuine change in update duration.
    history(path, counter=counter, slow=True)
    slow_run = load_run(path)
    assert assess_timing(slow_run).ratio == pytest.approx(3)
    finding, = ThroughputDetector().run(slow_run)
    assert finding.evidence["ratio"] == pytest.approx(3)
    assert finding.step_end == 1800


def test_logging_counter_does_not_prove_training_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "history.csv"
    history(path, counter=None)
    run = load_run(path)
    assert run.input_complete()
    assert run.axis_kind() == "logging_step"
    assert run.meta["step_column"] == "_step"
    assert ThroughputDetector().run(run) == []
    assert assess_timing(run).status == "unavailable"
    assert main(["analyze", str(path), "--output", "json", "--require-timing"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["run"]["step_axis"] == "logging_step"
    assert result["timing"]["ratio"] is None
    assert "W&B logging steps" in result["timing"]["reason"]
    html = render_html(run, [])
    assert "over logging steps 0 to 60" in html
    assert "<dt>logging steps</dt>" in html
    stream = StringIO()
    render_terminal(run, [], stream, color=False)
    assert "logging steps from _step" in stream.getvalue()


def test_direct_measurements_still_work_on_logging_axis(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    path.write_text("_step,_runtime,step_time\n" + "".join(
        f"{i},{i},{0.1 if i < 20 else 0.3}\n" for i in range(40)
    ))
    run = load_run(path)
    timing = assess_timing(run)
    assert timing.axis == "logging_step"
    assert timing.ratio == pytest.approx(3)
    assert timing.source == "step_time"
    findings = ThroughputDetector().run(run)
    assert len(findings) == 1
    assert "logging steps 32" in render_html(run, findings)
    assert "Logging steps</th>" in render_html(run, findings)


def test_multiple_training_axes_need_a_choice(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    path.write_text("_step,trainer/global_step,global_step,loss\n0,40,10,2\n1,80,20,1\n")
    with pytest.raises(ValueError, match="multiple W&B training-step columns"):
        load_run(path)
    chosen = load_run(path, step_key="trainer/global_step")
    assert chosen.get("train/loss").steps() == [40, 80]  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="not found"):
        load_run(path, step_key="typo")


@pytest.mark.parametrize("bad", ["", "1.5", "nan", "inf", "1e100", "true"])
def test_bad_training_counter_never_falls_back_to_logging_counter(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "history.csv"
    path.write_text(f"_step,trainer/global_step,loss\n0,10,2\n1,{bad},1\n2,30,0.5\n")
    run = load_run(path)
    assert run.get("train/loss").steps() == [10, 30]  # type: ignore[union-attr]
    assert run.skipped_rows() == 1
    assert not run.input_complete()


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "lightning", "wandb"])
def test_custom_counter_and_cli_watch_select_the_same_axis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fmt: str,
) -> None:
    path = tmp_path / "custom.csv"
    history(path, counter="optimizer/updates", slow=True)
    if fmt != "wandb":
        path.write_text(path.read_text().replace("_runtime", "wall_time"))
    if fmt == "jsonl":
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        path = path.with_suffix(".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = load_run(path, fmt=fmt, step_key="optimizer/updates")
    assert run.axis_kind() == "step"
    assert run.meta["step_column"] == "optimizer/updates"
    assert assess_timing(run).ratio == pytest.approx(3)
    args = [str(path), "--format", fmt, "--step-key", "optimizer/updates",
            "--fail-on", "warning", "--require-timing"]
    assert main(["analyze", *args, "--output", "json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["timing"]["ratio"] == pytest.approx(3)
    assert main(["watch", *args, "--once", "--no-color"]) == 1
    assert "steps" in capsys.readouterr().out


def test_explicit_wandb_counter_is_a_user_assertion(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    # The user knows their logger explicitly called log(..., step=optimizer_step).
    path.write_text("_step,_runtime,loss\n" + "".join(
        f"{i * 10},{i},1\n" for i in range(30)
    ))
    assert assess_timing(load_run(path)).status == "unavailable"
    asserted = load_run(path, step_key="_step")
    assert assess_timing(asserted).ratio == pytest.approx(1)
    assert json.loads(to_json_str(asserted, []))["run"]["step_axis"] == "step"


@pytest.mark.parametrize("axis,kind", [("Step", "logging_step"), ("_step", "logging_step"),
                                      ("trainer/global_step", "step"), ("epoch", "row_index")])
def test_chart_exports_distinguish_default_and_training_axes(
    tmp_path: Path, axis: str, kind: str,
) -> None:
    path = tmp_path / "chart.csv"
    path.write_text(f"{axis},run - loss,run - loss__MIN,run - loss__MAX\n10,2,1,3\n20,1,0,2\n")
    run = load_run(path)
    assert run.axis_kind() == kind
    with pytest.raises(ValueError, match="one shared x-axis"):
        load_run(path, step_key="absent")
    asserted = load_run(path, step_key=axis)
    assert asserted.axis_kind() == "step"
    assert asserted.get("train/loss").steps() == [10, 20]  # type: ignore[union-attr]


def test_sparse_custom_axis_cannot_impute_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "custom.jsonl"
    path.write_text('{"updates":10,"loss":2}\n{"loss":1.5}\n{"updates":30,"loss":1}\n')
    run = load_run(path, step_key="updates")
    assert run.get("train/loss").steps() == [10, 30]  # type: ignore[union-attr]
    assert run.skipped_rows() == 1
    assert not run.input_complete()


@pytest.mark.parametrize("fmt", ["hf", "tensorboard"])
def test_column_override_rejects_formats_without_columns(tmp_path: Path, fmt: str) -> None:
    with pytest.raises(ValueError, match="selects a column"):
        load_run(tmp_path / "unused", fmt=fmt, step_key="step")
