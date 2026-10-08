"""Counterexamples where reading count is not a usable training trajectory."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trainspotter.cli import main
from trainspotter.detectors import run_all
from trainspotter.detectors.divergence import DivergenceDetector
from trainspotter.detectors.plateau import PlateauDetector
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report import render_html, to_json_dict


def test_repeating_one_observation_cannot_create_a_plateau(tmp_path: Path) -> None:
    path = tmp_path / "repeated.csv"
    path.write_text("step,loss\n" + "10,1\n" * 100)
    run = load_run(path)
    assert len(run.metrics["train/loss"]) == 100  # original readings are retained
    assert run.input_complete()
    assert run_all(run) == []
    assert PlateauDetector().run(run) == []  # direct detector API is safe too
    report = to_json_dict(run, [])
    assert report["run"]["metrics"]["train/loss"]["analysis_count"] == 1
    assert report["run"]["repeated_steps"] == [
        {
            "metric": "train/loss",
            "step": 10,
            "count": 100,
            "conflicting_values": False,
            "conflicting_times": False,
            "readings": [{"value": 1.0, "wall_time": None, "epoch": None, "count": 100}],
        }
    ]
    html = render_html(run, [])
    assert 'class="marker-single"' in html
    assert "100 readings" in html
    assert "Equal readings: counted once" in html


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "hf"])
def test_conflicting_values_are_retained_but_never_become_a_guessed_curve(
    tmp_path: Path,
    fmt: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{"step": i, "loss": 1.0, "grad_norm": 1.0} for i in range(60)]
    rows.append({"step": 20, "loss": 20.0})
    path = tmp_path / ("input.json" if fmt == "hf" else f"input.{fmt}")
    if fmt == "csv":
        path.write_text(
            "step,loss,grad_norm\n"
            + "".join(f"{row['step']},{row['loss']},{row.get('grad_norm', '')}\n" for row in rows)
        )
    elif fmt == "jsonl":
        path.write_text("\n".join(json.dumps(row) for row in rows))
    else:
        path.write_text(json.dumps({"log_history": rows}))
    run = load_run(path)
    assert not run.input_complete()
    assert [p.value for p in run.metrics["train/loss"].points if p.step == 20] == [1, 20]
    assert "train/loss" not in run.analysis_view().metrics
    assert len(run.analysis_view().metrics["grad_norm"]) == 60
    assert not any(f.metric == "train/loss" for f in run_all(run))
    assert main(["analyze", str(path), "--output", "json", "--fail-on", "warning"]) == 2
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert not data["summary"]["input_complete"]
    assert data["run"]["metrics"]["train/loss"]["analysis_excluded"]
    assert data["run"]["repeated_steps"][0]["conflicting_values"]
    assert "input is incomplete" in output.err
    html = render_html(run, run_all(run))
    assert "Trajectory not plotted for train/loss" in html
    assert "Conflicting values: trajectory excluded" in html
    assert "Input incomplete" in html


def test_same_values_with_conflicting_wall_times_keep_value_checks_and_unknown_timing() -> None:
    run = Run()
    for i in range(30):
        run.add_point("train/loss", i, 1, float(i))
        run.add_point("train/loss", i, 1, 1000.0 + i)
    run.finalize()
    view = run.analysis_view()
    assert len(view.metrics["train/loss"]) == 30
    assert all(point.wall_time is None for point in view.metrics["train/loss"].points)
    assert not run.input_complete()
    assert not any(f.detector == "throughput" for f in run_all(run))
    assert "their timing is unknown" in " ".join(run.input_notes())


def test_repeated_nonfinite_values_are_observed_evidence_not_missing_values() -> None:
    run = Run()
    for step in range(50):
        for _ in range(3):
            run.add_point("train/loss", step, 1.0 if step < 40 else float("nan"))
    run.finalize()
    assert run.input_complete()
    assert math.isnan(run.analysis_view().metrics["train/loss"].points[-1].value)
    findings = DivergenceDetector().run(run)
    assert findings and findings[0].step_start == 40
    json.dumps(to_json_dict(run, findings), allow_nan=False)


def test_conflicts_do_not_hide_an_independent_metric_failure() -> None:
    run = Run()
    for step in range(50):
        run.add_point("train/loss", step, 1)
        run.add_point("eval/loss", step, 1 if step < 40 else float("inf"))
    run.add_point("train/loss", 20, 100)
    run.finalize()
    titles = [f.title for f in run_all(run) if f.metric == "eval/loss"]
    assert "Non-finite evaluation values" in titles
    assert not any(f.metric == "train/loss" for f in run_all(run))


def test_conflicting_trajectory_does_not_hide_nonfinite_readings_on_the_same_metric() -> None:
    run = Run()
    for step in range(50):
        run.add_point("train/loss", step, 1 if step < 40 else float("nan"))
    run.add_point("train/loss", 20, 100)
    run.add_point("train/loss", 40, 1)
    run.finalize()
    findings = run_all(run)
    assert len(findings) == 1
    assert findings[0].title == "Non-finite readings in ambiguous input"
    assert findings[0].step_start == 40
    assert findings[0].evidence["distinct_bad_steps"] == 10
    assert not run.input_complete()
    assert DivergenceDetector().run(run) == findings


def test_repeated_sparse_metric_rows_do_not_merge_different_metrics(tmp_path: Path) -> None:
    path = tmp_path / "lightning.csv"
    path.write_text("step,train_loss,eval_loss,epoch\n0,2,,0\n0,,3,0\n1,1,,0\n1,,2,0\n")
    run = load_run(path, fmt="lightning")
    assert run.input_complete()
    assert len(run.analysis_view().metrics["train/loss"]) == 2
    assert len(run.analysis_view().metrics["eval/loss"]) == 2
    assert len(run.analysis_view().metrics["epoch"]) == 2


def test_captured_event_files_retain_tensorboards_raw_readings() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    root = Path(__file__).resolve().parent.parent / "examples/repeated_steps"
    for name in ("replayed", "conflicting"):
        path = root / name
        run = load_run(path)
        oracle = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for tag, metric in (("loss", "train/loss"), ("grad_norm", "grad_norm")):
            expected = sorted([(p.step, p.value, p.wall_time) for p in oracle.Scalars(tag)])
            actual = sorted([(p.step, p.value, p.wall_time) for p in run.metrics[metric].points])
            assert actual == expected
        assert run.overwritten_points() == 0
        assert run.input_complete() == (name == "replayed")
        assert len(run.analysis_view().metrics["grad_norm"]) == 40
    replayed = load_run(root / "replayed")
    assert len(replayed.metrics["train/loss"]) == 120
    assert len(replayed.analysis_view().metrics["train/loss"]) == 40
