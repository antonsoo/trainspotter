"""Unreadable values are boundaries, not absent rows or observed NaN readings."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trainspotter.cli import _extends
from trainspotter.detectors import run_all
from trainspotter.detectors.divergence import DivergenceDetector
from trainspotter.detectors.grad_norm import GradNormDetector
from trainspotter.detectors.setbacks import SetbackDetector
from trainspotter.detectors.spikes import SpikeDetector
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report.json_report import to_json_dict
from trainspotter.timing import assess_timing


def _write(path: Path, values: list[object], fmt: str, metric: str = "loss") -> Path:
    rows = [{"step": i, metric: value} for i, value in enumerate(values)]
    if fmt == "hf":
        path = path.with_suffix(".json")
        path.write_text(json.dumps({"log_history": rows}))
    elif fmt == "jsonl":
        path = path.with_suffix(".jsonl")
        path.write_text("\n".join(json.dumps(row) for row in rows))
    elif fmt == "wandb":
        path = path.with_suffix(".csv")
        path.write_text(f"_step,trainer/global_step,{metric}\n" + "".join(
            f"{i},{i},{value}\n" for i, value in enumerate(values)
        ))
    elif fmt == "chart":
        path = path.with_suffix(".csv")
        path.write_text(f"Step,run - {metric},run - {metric}__MIN,run - {metric}__MAX\n" + "".join(
            f"{i},{value},,\n" for i, value in enumerate(values)
        ))
    else:
        path = path.with_suffix(".csv")
        path.write_text(f"step,{metric}\n" + "".join(
            f"{i},{value}\n" for i, value in enumerate(values)
        ))
    return path


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "hf", "wandb", "chart", "lightning"])
@pytest.mark.parametrize("kind", ["spike", "setback", "rise"])
def test_unreadable_values_do_not_join_a_trajectory(tmp_path: Path, fmt: str, kind: str) -> None:
    values: list[object]
    if kind == "spike":
        values = [5.0] * 20 + [1.0] * 40 + ["unreadable"] * 10 + [4.0] + [1.0] * 40
    elif kind == "setback":
        values = [5.0] * 20 + [1.0] * 40 + ["unreadable"] * 10 + [4.0] * 20
        values += ["unreadable"] * 10 + [1.0] * 40
    else:
        values = [1.0] * 50 + ["unreadable"] * 10 + [3.0] * 50
    run = load_run(_write(tmp_path / "run", values, fmt),
                   fmt="lightning" if fmt == "lightning" else None)
    series = run.get("train/loss")
    assert series is not None
    assert [p.step for p in series.points if p.value is None] == [
        i for i, value in enumerate(values) if value == "unreadable"
    ]
    assert not run.input_complete()
    for detector in [SpikeDetector(), SetbackDetector(), DivergenceDetector()]:
        assert detector.run(run) == []
    report = to_json_dict(run, run_all(run))
    assert report["run"]["metrics"]["train/loss"]["count"] == series.readable_count()
    assert len(report["run"]["metrics"]["train/loss"]["unreadable_steps"]) == run.skipped_values()


def test_a_real_recovered_event_after_an_unreadable_region_remains_local(tmp_path: Path) -> None:
    values: list[object] = [1.0] * 30 + ["broken"] * 10
    values += [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40
    run = load_run(_write(tmp_path / "run", values, "csv"))
    findings = SetbackDetector().run(run)
    assert [(f.step_start, f.step_end) for f in findings] == [(100, 120)]
    assert findings[0].evidence["segment_start_step"] == 40
    assert findings[0].evidence["steps_lost"] == 20
    assert "unreadable" in findings[0].message


def test_sparse_rows_and_text_metadata_do_not_become_broken_readings(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    path.write_text("step,loss,eval_loss,comment\n1,3,,start\n2,,2.5,evaluate\n3,2,,train\n")
    run = load_run(path)
    assert run.input_complete()
    assert run.get("train/loss").steps() == [1, 3]
    assert "comment" not in run.metrics
    assert not run._pending_unreadable


def test_custom_numeric_column_retains_unreadable_positions_before_its_first_number(tmp_path: Path):
    path = tmp_path / "run.csv"
    path.write_text("step,custom,comment\n1,broken,start\n2,1,note\n3,broken,end\n")
    run = load_run(path)
    assert [p.value for p in run.get("custom").points] == [None, 1.0, None]
    assert run.skipped_values() == 2
    assert "comment" not in run.metrics


@pytest.mark.parametrize("metric", ["train/loss", "grad_norm"])
def test_observed_nan_is_not_lost_or_invented_at_a_repeated_unreadable_position(metric: str):
    run = Run()
    run.add_point(metric, 1, 2.0)
    run.add_point(metric, 2, None)
    run.add_point(metric, 2, math.nan)
    run.add_point(metric, 3, 1.0)
    detector = DivergenceDetector() if metric == "train/loss" else GradNormDetector()
    findings = detector.run(run)
    assert len(findings) == 1
    assert findings[0].evidence["count"] == 1
    assert findings[0].step_start == findings[0].step_end == 2
    assert not run.repeated_steps()[metric][0].conflicting_values
    assert run.analysis_view().get(metric).points[1].value is None
    report = to_json_dict(run, findings)
    assert [r["value"] for r in report["run"]["repeated_steps"][0]["readings"]] == [None, "nan"]
    just_unreadable = Run()
    just_unreadable.add_point(metric, 2, None)
    assert detector.run(just_unreadable) == []
    assert not just_unreadable.input_complete()


def test_unreadable_timing_cell_cannot_shift_the_comparison_window(tmp_path: Path):
    values: list[object] = [1.0] * 20 + [2.0] * 9 + ["broken"]
    run = load_run(_write(tmp_path / "run", values, "csv", "step_time"))
    timing = assess_timing(run)
    assert timing.status == "unavailable"
    assert timing.recent.invalid_observations == 1
    assert timing.ratio is None


def test_watch_distinguishes_unreadable_values_from_observed_nan():
    previous = Run()
    current = Run()
    previous.add_point("train/loss", 1, None)
    current.add_point("train/loss", 1, math.nan)
    assert not _extends(previous, current)
    assert _extends(previous, previous)


def test_no_numeric_readings_is_still_an_input_error(tmp_path: Path, capsys):
    from trainspotter.cli import main

    path = _write(tmp_path / "run", ["broken"] * 5, "csv")
    assert main(["analyze", str(path), "--output", "json"]) == 2
    assert "no numeric metrics found" in capsys.readouterr().err


def test_html_download_retains_every_unreadable_position_without_observed_nan(tmp_path: Path):
    import base64
    import re

    from trainspotter.report.html_report import render_html

    values: list[object] = [1.0] * 20 + ["broken"] * 30 + [2.0] * 20
    run = load_run(_write(tmp_path / "run", values, "csv"))
    html = render_html(run, run_all(run))
    match = re.search(
        r'download="trainspotter-unreadable-values.json" '
        r'href="data:application/json;base64,([^"]+)"', html,
    )
    assert match is not None
    data = json.loads(base64.b64decode(match[1]))
    assert data["unreadable_steps"] == {"train/loss": list(range(20, 50))}
    assert "marker-unreadable" in html
    assert "Non-finite training loss" not in html
