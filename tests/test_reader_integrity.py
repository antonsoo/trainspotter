"""Coordinate and metric identity counterexamples, across the file-based readers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from trainspotter.readers import load_run
from trainspotter.readers.common import build_key_map, try_step


@pytest.mark.parametrize("bad", [None, True, False, "", "oops", "NaN", "inf", "1e400",
                                 "1.9", "-0.1", "9007199254740991.1", "9007199254740993",
                                 "1_000", "1e-999999999999999999999"])
@pytest.mark.parametrize("fmt", ["csv", "jsonl", "hf", "wandb", "lightning"])
def test_an_invalid_explicit_step_never_becomes_a_coordinate(
    tmp_path: Path, bad: object, fmt: str,
) -> None:
    if fmt in ("csv", "wandb", "lightning"):
        step = "_step" if fmt == "wandb" else "step"
        path = tmp_path / "run.csv"
        path.write_text(f"{step},loss\n100,2\n{'' if bad is None else bad},99\n200,1\n")
    else:
        rows = [{"step": 100, "loss": 2}, {"step": bad, "loss": 99}, {"step": 200, "loss": 1}]
        path = tmp_path / ("run.json" if fmt == "hf" else "run.jsonl")
        path.write_text(json.dumps({"log_history": rows}) if fmt == "hf" else
                        "\n".join(json.dumps(row) for row in rows))
    run = load_run(path, fmt=fmt)
    assert run.metrics["train/loss"].steps() == [100, 200]
    assert run.metrics["train/loss"].values() == [2, 1]
    assert run.skipped_rows() == 1
    assert not run.input_complete()


@pytest.mark.parametrize("step", [1.0, "1.0", "1e3", "+10", "-10", "9007199254740991"])
def test_exact_integral_steps_remain_supported(step: object) -> None:
    assert try_step(step) == int(float(step))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [2**53 + 1, 10**500, 1.5, True, float("nan")])
def test_numeric_steps_do_not_round_truncate_or_overflow(value: object) -> None:
    assert try_step(value) is None


@pytest.mark.parametrize("fmt", ["hf", "jsonl"])
def test_json_numeric_steps_are_validated_before_float_rounding(tmp_path: Path, fmt: str) -> None:
    rows = ['{"step":100,"loss":2}', '{"step":9007199254740991.1,"loss":9}',
            '{"step":200,"loss":1}']
    path = tmp_path / ("run.json" if fmt == "hf" else "run.jsonl")
    path.write_text('{"log_history":[' + ",".join(rows) + "]}" if fmt == "hf"
                    else "\n".join(rows))
    run = load_run(path)
    assert run.metrics["train/loss"].steps() == [100, 200]
    assert run.skipped_rows() == 1


def test_row_index_is_used_only_when_the_entire_source_has_no_step_column(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"loss":2}\n{"loss":1}\n')
    run = load_run(path)
    assert run.metrics["train/loss"].steps() == [0, 1]
    assert "not training-step counts" in " ".join(run.input_notes())
    path.write_text('{"step":100,"loss":2}\n{"loss":1}\n')
    run = load_run(path)
    assert run.metrics["train/loss"].steps() == [100]
    assert run.skipped_rows() == 1


def test_normalization_keeps_distinct_metrics_for_every_input_order(tmp_path: Path) -> None:
    keys = ["eval_accuracy", "eval/accuracy", "loss", " loss ", "loss ", "train/loss"]
    for order in itertools.permutations(keys):
        names = build_key_map(order)
        assert len(set(names.values())) == len(keys), names
        assert names["train/loss"] == "train/loss"
        assert names["eval/accuracy"] == "eval/accuracy"
    path = tmp_path / "run.csv"
    path.write_text("step," + ",".join(keys) + "\n10,1,2,3,4,5,6\n")
    run = load_run(path)
    assert len(run.metrics) == len(keys)
    assert sorted(series.values()[0] for series in run.metrics.values()) == [1, 2, 3, 4, 5, 6]
    assert set(run.meta["metric_sources"].values()) == set(keys)


@pytest.mark.parametrize("fmt,header", [
    ("csv", "step,loss,loss"), ("lightning", "step,loss,loss"),
    ("wandb", "_step,loss,loss"), ("wandb", "Step,a - loss,a - loss"),
])
def test_duplicate_csv_headers_are_rejected(tmp_path: Path, fmt: str, header: str) -> None:
    path = tmp_path / "run.csv"
    path.write_text(header + "\n1,2,9\n")
    with pytest.raises(ValueError, match="duplicate CSV"):
        load_run(path, fmt=fmt)


def test_duplicate_json_keys_do_not_silently_hide_a_reading(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"step":1,"loss":2}\n{"step":2,"loss":99,"loss":1}\n')
    run = load_run(path)
    assert run.metrics["train/loss"].values() == [2]
    assert run.skipped_rows() == 1
    state = tmp_path / "run.json"
    state.write_text('{"log_history":[{"step":2,"loss":99,"loss":1}]}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_run(state)


def test_multiple_step_axes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    path.write_text("step,Step,loss\n1,900,2\n")
    with pytest.raises(ValueError, match="multiple step columns"):
        load_run(path)


def test_wandb_chart_does_not_invent_missing_steps(tmp_path: Path) -> None:
    path = tmp_path / "chart.csv"
    path.write_text("Step,a - loss,a - loss__MIN,a - loss__MAX\n100,2,2,2\n,99,99,99\n")
    run = load_run(path)
    assert run.metrics["train/loss"].steps() == [100]
    assert run.skipped_rows() == 1


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_wandb_history_spreadsheet_delimiters(tmp_path: Path, delimiter: str) -> None:
    path = tmp_path / "wandb.csv"
    path.write_text(f"_step{delimiter}loss\n100{delimiter}2,5\n200{delimiter}1,25\n")
    run = load_run(path)
    assert run.source_format == "wandb"
    assert run.metrics["train/loss"].steps() == [100, 200]
    assert run.metrics["train/loss"].values() == [2.5, 1.25]


def test_huge_integer_metrics_remain_nonfinite_evidence(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"step": 1, "loss": 10**500}))
    run = load_run(path)
    assert run.metrics["train/loss"].values() == [float("inf")]


def test_invalid_metric_cells_are_visible_but_sparse_and_text_columns_are_allowed(tmp_path: Path):
    path = tmp_path / "run.csv"
    path.write_text("step,loss,custom,comment\n1,2,1,started\n2,broken,broken,note\n3,,,sparse\n")
    run = load_run(path)
    assert run.skipped_values() == 2
    assert not run.input_complete()
    path.write_text("step,loss,comment\n1,2,started\n2,,sparse\n3,nan,diverged\n")
    run = load_run(path)
    assert run.skipped_values() == 0
    assert run.input_complete()


def test_trainer_metadata_cannot_forge_reader_accounting(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "skipped_rows": -1, "selected_run": "invented", "restart_markers": 9,
        "log_history": [{"step": 1, "loss": 2}, {"step": "bad", "loss": 1}],
    }))
    run = load_run(path)
    assert run.skipped_rows() == 1
    assert not run.input_complete()
    assert "selected_run" not in run.meta and "restart_markers" not in run.meta


def test_wandb_spaced_bookkeeping_headers_keep_the_real_step_axis(tmp_path: Path):
    path = tmp_path / "run.csv"
    path.write_text(" _step , _runtime , loss \n100,1,2\n200,2,1\n")
    run = load_run(path)
    assert run.source_format == "wandb"
    assert run.metrics["train/loss"].steps() == [100, 200]
    assert run.metrics["train/loss"].points[0].wall_time == 1
