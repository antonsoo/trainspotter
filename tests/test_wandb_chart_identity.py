"""Chart labels do not encode which separators belong to run or metric names."""

import csv
import json
from html import escape

import pytest

from trainspotter import cli
from trainspotter.readers import load_run
from trainspotter.readers.wandb import read_wandb_chart_csv, read_wandb_csv
from trainspotter.report.html_report import render_html


def chart(path, columns, count=50):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trainer/global_step", *[
            label + suffix for label in columns for suffix in ("", "__MIN", "__MAX")
        ]])
        for step in range(count):
            writer.writerow([step, *[
                value(step) for value in columns.values() for _ in range(3)
            ]])
    return path


@pytest.mark.parametrize("reader", [load_run, read_wandb_csv, read_wandb_chart_csv])
@pytest.mark.parametrize("selected", [None, "runA", "runA - eval"])
def test_ambiguous_labels_cannot_silently_select_or_relabel_loss(tmp_path, reader, selected):
    path = chart(tmp_path / "chart.csv", {
        "runA - train - loss": lambda i: 2 - i * .02,
        "runA - eval - loss": lambda i: .5 + i * .02,
    })
    with pytest.raises(ValueError, match="ambiguous W&B chart label"):
        reader(path, run_name=selected)
    run = reader(path, chart_single_run="runA")
    assert set(run.metrics) == {"train/loss", "eval/loss"}
    assert run.metrics["train/loss"].values() == [2 - i * .02 for i in range(50)]
    assert run.metrics["eval/loss"].values() == [.5 + i * .02 for i in range(50)]
    assert run.meta["selected_run"] == "runA"
    assert "runs" not in run.meta
    assert run.input_complete()


@pytest.mark.parametrize("name", ["runA - resumed", "<run & \"name\">", " runA "])
def test_exact_declared_name_is_preserved_in_reports(tmp_path, capsys, name):
    label = name + " - eval - loss"
    path = chart(tmp_path / "chart.csv", {label: lambda i: 1 + i})
    assert cli.main(["analyze", str(path), "--chart-single-run", name,
                     "--output", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["run"]["metric_sources"] == {"eval/loss": "eval - loss"}
    assert report["run"]["chart_identity"] == {
        "interpretation": "declared_single_run", "selected_run": name,
        "columns": {"eval - loss": label},
    }
    assert "CSV alone cannot verify" in " ".join(report["run"]["input_notes"])
    html = render_html(load_run(path, chart_single_run=name), [])
    assert escape(label) in html
    assert "CSV alone cannot verify" in html


def test_ordinary_multi_run_selection_still_works(tmp_path):
    path = chart(tmp_path / "chart.csv", {
        "runA - train/loss": lambda i: 2 - i * .02,
        "runB - eval/loss": lambda i: .5 + i * .02,
    })
    assert load_run(path).meta["selected_run"] == "runA"
    assert set(load_run(path, run_name="runB").metrics) == {"eval/loss"}
    with pytest.raises(ValueError, match="start of 2 runs"):
        load_run(path, run_name="run")
    with pytest.raises(ValueError, match="does not start with"):
        load_run(path, chart_single_run="runA")


def test_multi_run_chart_with_separator_names_requires_a_different_export(tmp_path):
    path = chart(tmp_path / "chart.csv", {
        "sweep - lr 3e-4 - train/loss": lambda i: 1,
        "sweep - lr 1e-4 - eval/loss": lambda i: 2,
    })
    # The old last-separator heuristic interpreted this layout, but cannot prove it.
    with pytest.raises(ValueError, match="ambiguous W&B chart label"):
        load_run(path, run_name="sweep - lr 3e-4")
    with pytest.raises(ValueError, match="does not start with"):
        load_run(path, chart_single_run="sweep - lr 3e-4")


@pytest.mark.parametrize("name", ["run", "", " "])
def test_declaration_is_not_a_run_name_prefix(tmp_path, name):
    path = chart(tmp_path / "chart.csv", {"runA - loss": lambda i: 1})
    with pytest.raises(ValueError, match="does not start with|nonblank"):
        load_run(path, chart_single_run=name)


def test_unrecognized_value_columns_must_also_belong_to_declared_run(tmp_path):
    path = chart(tmp_path / "chart.csv", {
        "runA - loss": lambda i: 1, "runB - custom": lambda i: "metadata",
    })
    with pytest.raises(ValueError, match="runB - custom"):
        load_run(path, chart_single_run="runA")


@pytest.mark.parametrize("header", ["step,loss", "_step,loss"])
def test_declaration_is_rejected_on_other_csv_formats(tmp_path, header):
    path = tmp_path / "history.csv"
    path.write_text(header + "\n0,1\n")
    with pytest.raises(ValueError, match="applies only to W&B chart"):
        load_run(path, chart_single_run="runA")


def test_declaration_and_run_selection_are_mutually_exclusive(tmp_path):
    path = chart(tmp_path / "chart.csv", {"runA - loss": lambda i: 1})
    with pytest.raises(ValueError, match="cannot be combined"):
        load_run(path, run_name="runA", chart_single_run="runA")
    for command in ("analyze", "watch"):
        with pytest.raises(SystemExit) as exc:
            cli.main([command, str(path), "--run", "runA", "--chart-single-run", "runA"])
        assert exc.value.code == 2


def test_empty_and_colliding_chart_metrics_are_rejected(tmp_path):
    path = chart(tmp_path / "chart.csv", {"runA - ": lambda i: 1})
    with pytest.raises(ValueError, match="empty metric name"):
        load_run(path, chart_single_run="runA")
    chart(path, {"loss": lambda i: 1, " - loss": lambda i: 2})
    with pytest.raises(ValueError, match="multiple chart columns"):
        load_run(path)


def test_watch_reloads_hold_declaration_and_reject_an_added_run(tmp_path, monkeypatch, capsys):
    columns = {"runA - train - loss": lambda i: 2 - i * .02,
               "runA - eval - loss": lambda i: .5 + i * .02}
    path = chart(tmp_path / "chart.csv", columns)
    actions = iter([
        lambda: chart(path, columns, count=51),
        lambda: chart(path, {**columns, "runB - loss": lambda i: 1}),
        lambda: chart(path, columns, count=52),
    ])

    def sleep(_):
        action = next(actions, None)
        if action is None:
            raise KeyboardInterrupt
        action()

    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli.main(["watch", str(path), "--chart-single-run", "runA"]) == 0
    output = capsys.readouterr().out
    assert output.count("selection held while watching") == 1
    assert output.count("skipped this read") == 1
    assert "runB - loss" in output
    assert "cannot be combined" not in output


def test_watch_treats_changed_source_metric_as_a_rewrite(tmp_path):
    path = tmp_path / "chart.csv"
    before = load_run(chart(path, {"runA - loss": lambda i: 1}), chart_single_run="runA")
    after = load_run(chart(path, {"runA - train/loss": lambda i: 1}), chart_single_run="runA")
    assert not cli._extends(before, after)
