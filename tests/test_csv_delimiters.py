"""Experiment labels cannot change the delimiter, coordinates or observations."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.readers.tabular import _delimiter


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
@pytest.mark.parametrize(
    "name",
    [
        "run",
        "seed=0; lr=0.001; wd=0.01",
        "seed=0\tlr=0.001\twd=0.01",
        'seed=0, note="main"; lr=0.001; wd=0.01',
        "seed=0; lr=0.001;\nwd=0.01",
    ],
)
def test_csv_writer_chart_labels_preserve_all_points_and_findings(
    tmp_path: Path, delimiter: str, name: str
) -> None:
    path = tmp_path / "chart.csv"
    train = [2 - step * 0.02 for step in range(50)]
    evaluation = [0.5 + step * 0.02 for step in range(50)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(
            [
                "global_step",
                *[
                    f"{name} - {metric}{suffix}"
                    for metric in ["loss", "eval_loss"]
                    for suffix in ["", "__MIN", "__MAX"]
                ],
            ]
        )
        writer.writerows([[i, *[train[i]] * 3, *[evaluation[i]] * 3] for i in range(50)])
    run = load_run(path)
    assert run.source_format == "wandb"
    assert run.meta["step_column"] == "global_step"
    assert run.meta["selected_run"] == name
    assert [(p.step, p.value) for p in run.get("train/loss").points] == list(enumerate(train))
    assert [(p.step, p.value) for p in run.get("eval/loss").points] == list(enumerate(evaluation))
    assert [finding.detector for finding in run_all(run)] == ["overfitting"]
    assert run.input_complete()


def test_malformed_rows_cannot_replace_a_known_axis_with_punctuation_fragments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss", "comment;meta"])
        writer.writerow([0, 1, "note;7"])
        writer.writerow([1, 2, "note;8", "extra"])
        writer.writerow([2, 3, "note;9"])
    run = load_run(path)
    assert run.meta["step_column"] == "step"
    assert [(p.step, p.value) for p in run.get("train/loss").points] == [(0, 1), (1, None), (2, 3)]
    assert run.skipped_rows() == 1
    assert not run.input_complete()


def test_an_explicit_coordinate_column_constrains_format_detection_too(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text('"optimizer\nupdates",loss;extra\n0,1\n1,2\n')
    run = load_run(path, step_key="optimizer\nupdates")
    assert run.meta["step_column"] == "optimizer\nupdates"
    assert [(p.step, p.value) for p in run.get("loss;extra").points] == [(0, 1), (1, 2)]


def test_an_axis_word_inside_a_label_needs_matching_records(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("note;step;extra,loss\nrecorded,1\nrecorded,2\n")
    run = load_run(path)
    assert run.meta["step_column"] == "(row index)"
    assert [(p.step, p.value) for p in run.get("train/loss").points] == [(0, 1), (1, 2)]


def test_two_consistent_interpretations_require_unambiguous_quoting(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("first,second;third\n1,2;3\n4,5;6\n")
    with pytest.raises(ValueError, match="ambiguous CSV delimiter"):
        load_run(path)
    with path.open("w", newline="") as stream:
        csv.writer(stream, quoting=csv.QUOTE_ALL).writerows(
            [["first", "second;third"], [1, "2;3"], [4, "5;6"]]
        )
    run = load_run(path)
    assert run.meta["columns"] == ["first", "second;third"]
    assert [(p.step, p.value) for p in run.get("first").points] == [(0, 1), (1, 4)]


def test_probing_rewinds_and_does_not_read_an_entire_long_history() -> None:
    class BoundedSource(io.StringIO):
        def __next__(self) -> str:
            assert self.tell() < 1000, "delimiter inference consumed the full history"
            return super().__next__()

    source = BoundedSource("step,loss\n" + "0,1\n" * 10_000)
    assert _delimiter(source) == ","
    assert source.tell() == 0
