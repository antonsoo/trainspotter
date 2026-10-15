"""A rejected JSONL record still has an ordinal source position, never a step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainspotter.cli import _extends
from trainspotter.detectors import run_all
from trainspotter.readers import load_run


@pytest.mark.parametrize("bad", [
    '{"loss":', '{"loss":99,"loss":1}', '[{"loss":99}]', 'null', 'false', '"text"',
    '9', '{"loss":99} trailing', '[' * 2000,
])
def test_rejected_records_keep_positions_without_trusting_their_contents(tmp_path: Path, bad: str):
    path = tmp_path / "run.jsonl"
    path.write_text('{"loss":2,"custom":7,"comment":"start"}\n' + bad
                    + '\n{"loss":1,"custom":8,"comment":"end"}\n')
    run = load_run(path)
    for name, values in [("train/loss", [2, None, 1]), ("custom", [7, None, 8])]:
        assert [(p.step, p.value, p.wall_time) for p in run.metrics[name].points] == [
            (i, value, None) for i, value in enumerate(values)]
    assert "comment" not in run.metrics
    assert run.axis_kind() == "row_index"
    assert run.skipped_rows() == 1 and run.skipped_values() == 0
    assert not run.input_complete()
    assert not any(f.title == "Non-finite training loss" for f in run_all(run))


@pytest.mark.parametrize("local_event", [False, True])
def test_trajectories_stop_at_rejected_records_but_later_events_survive(
    tmp_path: Path, local_event: bool,
):
    values = ([1.] * 30 + [None] * 10 + [5.] * 20 + [1.] * 40 + [4.] * 20 + [1.] * 40
              if local_event else [1.] * 50 + [None] * 10 + [3.] * 50)
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(
        '{"loss":' if value is None else json.dumps({"loss": value}) for value in values
    ) + "\n")
    run = load_run(path)
    targets = [f for f in run_all(run) if f.detector in {"divergence", "spikes", "setbacks"}]
    assert [(f.title, f.step_start, f.step_end) for f in targets] == (
        [("Setback", 100, 120)] if local_event else [])
    if local_event:
        assert targets[0].evidence["segment_start_step"] == 40


def test_leading_trailing_records_and_late_metrics_keep_all_source_positions(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text('broken\n{"loss":1}\nnull\n{"custom":3}\n[4]\n')
    run = load_run(path)
    assert [(p.step, p.value) for p in run.metrics["train/loss"].points] == [
        (0, None), (1, 1), (2, None), (4, None)]
    assert [(p.step, p.value) for p in run.metrics["custom"].points] == [
        (0, None), (2, None), (3, 3), (4, None)]
    assert run.skipped_rows() == 3


@pytest.mark.parametrize("step_key", [None, "update"])
def test_an_explicit_axis_never_borrows_a_position_from_rejected_json(
    tmp_path: Path, step_key: str | None,
):
    key = step_key or "step"
    path = tmp_path / "run.jsonl"
    path.write_text('broken\n' + json.dumps({key: 100, "loss": 2}) + '\n'
                    + json.dumps({key: 150, "loss": 99})[:-1] + '\n'
                    + json.dumps({key: 200, "loss": 1}) + '\n')
    run = load_run(path, step_key=step_key)
    assert [(p.step, p.value) for p in run.metrics["train/loss"].points] == [(100, 2), (200, 1)]
    assert run.axis_kind() == "step"
    assert run.skipped_rows() == 2


def test_sparse_objects_and_ignored_blank_lines_do_not_become_rejections(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text('\ufeff\r\n{"loss":1}\r\n \t\r\n{"loss":null}\r\n{}\r\n'
                    '{"loss":""}\r\n{"loss":2}\r\n', encoding="utf-8")
    run = load_run(path)
    assert [(p.step, p.value) for p in run.metrics["train/loss"].points] == [(0, 1), (4, 2)]
    assert run.input_complete()


def test_completing_a_partial_last_record_replaces_its_boundary(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"loss":1}\n{"loss":')
    previous = load_run(path)
    assert [(p.step, p.value) for p in previous.metrics["train/loss"].points] == [
        (0, 1), (1, None)]
    with path.open("a") as writer:
        writer.write("2}\n")
    current = load_run(path)
    assert [(p.step, p.value) for p in current.metrics["train/loss"].points] == [(0, 1), (1, 2)]
    assert current.input_complete()
    assert not _extends(previous, current)
