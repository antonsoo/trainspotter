"""A rejected CSV record can still locate an unknown observation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainspotter.cli import _extends
from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report.json_report import to_json_dict


def write_case(path: Path, values: list[float | None], fmt: str, shape: str) -> None:
    headers = {
        "csv": "step,loss,comment",
        "lightning": "step,train_loss_step,comment",
        "wandb": "_step,trainer/global_step,loss,comment",
        "chart": "Step,run - loss,run - loss__MIN,run - loss__MAX",
        "row-index": "loss,comment",
    }
    rows = []
    for step, value in enumerate(values):
        cells = [str(step)] if fmt != "row-index" else []
        if fmt == "wandb":
            cells.append(str(step))
        if value is None and shape == "short":
            # A row-index file needs one field so DictReader retains the record.
            if not cells:
                cells = ["unreadable"]
        else:
            cells += [str(value) if value is not None else "unreadable", "note"]
            if fmt == "chart":
                cells[-1:] = ["", ""]
            if value is None and shape == "extra":
                cells.append("extra")
        rows.append(",".join(cells))
    path.write_text(headers[fmt] + "\n" + "\n".join(rows) + "\n")


@pytest.mark.parametrize("fmt", ["csv", "lightning", "wandb", "chart", "row-index"])
@pytest.mark.parametrize("shape", ["short", "extra"])
@pytest.mark.parametrize("kind", ["rise", "spike", "setback", "local"])
def test_bad_width_retains_gap_without_inventing_or_hiding_local_events(
    tmp_path: Path,
    fmt: str,
    shape: str,
    kind: str,
) -> None:
    values: list[float | None]
    if kind == "rise":
        values = [1.0] * 50 + [None] * 10 + [3.0] * 50
    elif kind == "spike":
        values = [5.0] * 20 + [1.0] * 40 + [None] * 10 + [4.0] + [1.0] * 40
    elif kind == "setback":
        values = [5.0] * 20 + [1.0] * 40 + [None] * 10 + [4.0] * 20 + [None] * 10 + [1.0] * 40
    else:
        values = [1.0] * 30 + [None] * 10 + [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40
    path = tmp_path / "metrics.csv"
    write_case(path, values, fmt, shape)
    run = load_run(path, fmt="lightning" if fmt == "lightning" else None)
    series = run.get("train/loss")
    assert series is not None
    gaps = [step for step, value in enumerate(values) if value is None]
    assert [p.step for p in series.points if p.value is None] == gaps
    assert series.readable_count() == len(values) - len(gaps)
    assert run.skipped_rows() == len(gaps)
    assert run.skipped_values() == 0  # Row shape, not a successfully located invalid scalar.
    assert not run.input_complete()
    targets = [f for f in run_all(run) if f.detector in {"divergence", "spikes", "setbacks"}]
    if kind == "local":
        assert [(f.title, f.step_start, f.step_end) for f in targets] == [("Setback", 100, 120)]
        assert targets[0].evidence["segment_start_step"] == 40
    else:
        assert targets == []
    report = to_json_dict(run, targets)
    assert report["run"]["metrics"]["train/loss"]["unreadable_steps"] == gaps


def test_bad_width_does_not_trust_values_or_timestamps_even_if_they_parse(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    path.write_text("step,loss,wall_time\n0,1,100\n1,nan,101,extra\n2,1,102\n")
    run = load_run(path)
    point = run.get("train/loss").points[1]
    assert point.step == 1 and point.value is None and point.wall_time is None
    assert not any(f.title == "Non-finite training loss" for f in run_all(run))


@pytest.mark.parametrize("bad_step", ["", "oops", "NaN", "1.5"])
def test_bad_width_cannot_invent_an_explicit_step(tmp_path: Path, bad_step: str):
    path = tmp_path / "metrics.csv"
    path.write_text(f"step,loss,comment\n0,1,ok\n{bad_step},broken\n2,1,ok\n")
    run = load_run(path)
    assert run.get("train/loss").steps() == [0, 2]
    assert run.skipped_rows() == 1


def test_sparse_complete_rows_remain_absent_and_custom_metrics_keep_gaps(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "step,loss,custom,comment\n0,,1,start\n1,2,,sparse\n2,broken,broken\n3,1,2,end\n"
    )
    run = load_run(path)
    assert [(p.step, p.value) for p in run.get("train/loss").points] == [
        (1, 2.0),
        (2, None),
        (3, 1.0),
    ]
    assert [(p.step, p.value) for p in run.get("custom").points] == [(0, 1.0), (2, None), (3, 2.0)]
    assert "comment" not in run.metrics


def test_watch_rewrite_replaces_an_unknown_row_with_the_actual_reading(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    path.write_text("step,loss,comment\n0,1,start\n1,2\n")
    previous = load_run(path)
    assert previous.get("train/loss").points[-1].value is None
    path.write_text("step,loss,comment\n0,1,start\n1,2,finished\n")
    current = load_run(path)
    assert current.get("train/loss").points[-1].value == 2
    assert current.input_complete()
    assert not _extends(previous, current)
    assert json.dumps(to_json_dict(current, run_all(current)), allow_nan=False)


@pytest.mark.parametrize("shape", ["short", "extra"])
def test_chart_time_axis_keeps_ordinal_gap_without_a_trusted_clock(tmp_path: Path, shape: str):
    path = tmp_path / "chart.csv"
    broken = "101" if shape == "short" else "101,2,2,2,extra"
    path.write_text(
        "Time,a - loss,a - loss__MIN,a - loss__MAX\n100,1,1,1\n" + broken + "\n102,1,1,1\n"
    )
    run = load_run(path)
    assert run.axis_kind() == "row_index"
    assert [(p.step, p.value, p.wall_time) for p in run.get("train/loss").points] == [
        (0, 1.0, 100.0),
        (1, None, None),
        (2, 1.0, 102.0),
    ]


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_decimal_comma_and_explicit_axis_keep_a_later_numeric_metrics_gap(
    tmp_path: Path,
    delimiter: str,
):
    path = tmp_path / "export.csv"
    rows = ["update;custom;comment", "100;;start", "101;2,5", "102;1,5;end"]
    path.write_text("\ufeff" + "\n".join(row.replace(";", delimiter) for row in rows) + "\n")
    run = load_run(path, step_key="update")
    assert [(p.step, p.value) for p in run.get("custom").points] == [(101, None), (102, 1.5)]
    assert "comment" not in run.metrics
    assert run.skipped_rows() == 1
