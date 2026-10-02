"""The same run, as different tools write it down.

A training run reaches trainspotter through whatever logged it: `transformers.Trainer`
sends `train/learning_rate` to W&B and TensorBoard where its own `trainer_state.json`
says `learning_rate`; Keras puts the validation curve in a directory of its own; a W&B
chart export holds every run that was drawn in the chart. Each test here writes one of
the bundled example runs in such a form and asks for the findings the original gives.

The layouts are the ones those tools produce: the W&B chart export's columns were
checked against exports committed to public repositories (`"Step","<run> - <metric>",
"<run> - <metric>__MIN","<run> - <metric>__MAX"`), and the event files are written by
TensorBoard's own writer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trainspotter.cli import main
from trainspotter.detectors import Finding, run_all
from trainspotter.model import Run
from trainspotter.readers import detect_format, load_run
from trainspotter.readers.common import build_key_map, normalize_key

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# trainer_state.json key -> the name transformers.Trainer logs it under in W&B and TensorBoard.
HF_REPORTED = {
    "loss": "train/loss",
    "learning_rate": "train/learning_rate",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
    "eval_loss": "eval/loss",
}


def _points(name: str) -> dict[str, list[tuple[int, float]]]:
    """The example run as {trainer_state key: [(step, value), ...]}, summary entry left out."""
    history = json.loads((EXAMPLES / f"{name}.trainer_state.json").read_text())["log_history"]
    out: dict[str, list[tuple[int, float]]] = {}
    for entry in history:
        if "train_runtime" in entry:
            continue
        for key, value in entry.items():
            if key != "step" and isinstance(value, int | float):
                out.setdefault(key, []).append((entry["step"], float(value)))
    return out


def _shape(findings: list[Finding]) -> list[tuple[str, str, str, int, int]]:
    return sorted((f.detector, f.severity, f.metric, f.step_start, f.step_end) for f in findings)


def _reference(name: str) -> list[tuple[str, str, str, int, int]]:
    return _shape(run_all(load_run(EXAMPLES / f"{name}.trainer_state.json")))


def _write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)


def _table(
    series: dict[str, list[tuple[int, float]]],
) -> tuple[list[int], dict[str, dict[int, float]]]:
    steps = sorted({step for points in series.values() for step, _ in points})
    return steps, {name: dict(points) for name, points in series.items()}


EXAMPLE_NAMES = ["divergence", "overfitting", "missing_warmup", "baseline"]


# ------------------------------------------------------------------ metric names


def test_names_transformers_reports_to_wandb_and_tensorboard() -> None:
    names = build_key_map(
        [*HF_REPORTED.values(), "train/global_step", "eval/runtime", "eval/accuracy"]
    )
    assert names == {
        "train/loss": "train/loss",
        "train/learning_rate": "lr",
        "train/grad_norm": "grad_norm",
        "train/epoch": "epoch",
        "eval/loss": "eval/loss",
        "train/global_step": "train/global_step",
        "eval/runtime": "eval/runtime",
        "eval/accuracy": "eval/accuracy",
    }


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("loss", "train/loss"),
        ("train_loss", "train/loss"),
        ("Loss/train", "train/loss"),
        ("train_loss_step", "train/loss"),
        ("batch_loss", "train/loss"),
        ("epoch_loss", "train/loss"),
        ("val_loss", "eval/loss"),
        ("val/loss", "eval/loss"),
        ("valid_loss", "eval/loss"),
        ("validation/loss", "eval/loss"),
        ("Loss/val", "eval/loss"),
        ("dev_loss", "eval/loss"),
        ("validation/epoch_loss", "eval/loss"),
        ("learning_rate", "lr"),
        ("train/lr", "lr"),
        ("lr-AdamW", "lr"),
        ("lr-SGD/pg1", "lr"),
        ("lr/pg0", "lr"),
        ("epoch_learning_rate", "lr"),
        ("train/grad_norm", "grad_norm"),
        ("gradient_norm", "grad_norm"),
        ("grad_norm/global", "grad_norm"),
        ("train_samples_per_second", "throughput"),
        ("eval_accuracy", "eval/accuracy"),
    ],
)
def test_recognized_names(raw: str, canonical: str) -> None:
    assert normalize_key(raw) == canonical


@pytest.mark.parametrize(
    "raw",
    [
        "val/box_loss",  # a component of the loss, not the loss
        "train/cls_loss",
        "loss_scale",
        "lr_scheduler_steps",
        "weird metric",
        "test/loss",  # the test set is not what a run is validated on while it trains
        "learning_rate_decay",
        "metrics/mAP50(B)",
    ],
)
def test_names_that_only_look_alike_are_left_alone(raw: str) -> None:
    assert normalize_key(raw) == raw


def test_one_metric_per_canonical_name() -> None:
    # Lightning: self.log("train_loss", on_step=True, on_epoch=True), a per-group LR monitor.
    lightning = build_key_map(
        [
            "epoch",
            "step",
            "train_loss_epoch",
            "train_loss_step",
            "val_loss",
            "lr-AdamW/pg2",
            "lr-AdamW/pg1",
        ]
    )
    assert lightning["train_loss_step"] == "train/loss"
    assert lightning["train_loss_epoch"] == "train_loss_epoch"
    assert lightning["lr-AdamW/pg1"] == "lr"
    assert lightning["lr-AdamW/pg2"] == "lr-AdamW/pg2"
    assert len(set(lightning.values())) == len(lightning)
    # The exact canonical name always wins, wherever it stands, and a loser keeps its own name.
    assert build_key_map(["loss", "train/loss"]) == {"loss": "loss", "train/loss": "train/loss"}
    assert build_key_map(["eval_loss", "eval/loss"]) == {
        "eval_loss": "eval_loss",
        "eval/loss": "eval/loss",
    }
    # Two names of equal standing: the first.
    both = build_key_map(["train_samples_per_second", "samples_per_second"])
    assert sorted(both.values()) == ["throughput", "train_samples_per_second"]


def test_a_lightning_log_with_step_and_epoch_losses_is_one_loss_curve(tmp_path: Path) -> None:
    series = _points("overfitting")
    steps, by = _table(
        {
            "train_loss_step": series["loss"],
            "val_loss": series["eval_loss"],
            "lr-AdamW": series["learning_rate"],
        }
    )
    epoch_means = {
        step: by["train_loss_step"][step] for step in steps[9::10] if step in by["train_loss_step"]
    }
    header = ["epoch", "step", "train_loss_step", "val_loss", "lr-AdamW", "train_loss_epoch"]
    rows = [
        [
            step // 100,
            step,
            by["train_loss_step"].get(step, ""),
            by["val_loss"].get(step, ""),
            by["lr-AdamW"].get(step, ""),
            epoch_means.get(step, ""),
        ]
        for step in steps
    ]
    path = tmp_path / "metrics.csv"
    _write_rows(path, header, rows)
    run = load_run(path)
    assert {"train/loss", "eval/loss", "lr", "train_loss_epoch"} <= set(run.metric_names())
    assert len(run.get("train/loss")) == len(series["loss"])  # type: ignore[arg-type]
    assert _shape(run_all(run)) == _reference("overfitting")


# ------------------------------------------------------------------ W&B


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_wandb_history_of_a_transformers_run(tmp_path: Path, name: str) -> None:
    """`wandb.Api().run(...).history()` of a run logged by transformers' W&B callback."""
    series = {
        HF_REPORTED[key]: points for key, points in _points(name).items() if key in HF_REPORTED
    }
    steps, by = _table(series)
    header = ["", "_step", "_runtime", "_timestamp", *series, "train/global_step"]
    rows = [
        [
            i,
            step,
            2.0 * i,
            1_790_000_000 + 2 * i,
            *(by[metric].get(step, "") for metric in series),
            step,
        ]
        for i, step in enumerate(steps)
    ]
    path = tmp_path / "history.csv"
    _write_rows(path, header, rows)
    assert detect_format(path) == "wandb"
    run = load_run(path)
    assert {"train/loss", "lr"} <= set(run.metric_names())
    assert "train/learning_rate" not in run.metric_names()
    assert _shape(run_all(run)) == _reference(name)


def _chart_export(
    path: Path, runs: dict[str, dict[str, list[tuple[int, float]]]], axis: str = "Step"
) -> None:
    """A chart's "Export > CSV": the axis, then for each run and metric the value and its band."""
    steps = sorted(
        {step for series in runs.values() for points in series.values() for step, _ in points}
    )
    header = [axis]
    columns: list[dict[int, float]] = []
    for run_name, series in runs.items():
        for metric, points in series.items():
            header += [
                f"{run_name} - {metric}",
                f"{run_name} - {metric}__MIN",
                f"{run_name} - {metric}__MAX",
            ]
            columns.append(dict(points))
    rows = []
    for i, step in enumerate(steps):
        row: list[object] = [step if axis == "Step" else round(1.5 * i, 3)]
        for column in columns:
            value = column.get(step, "")
            row += [
                value,
                value if value == "" else value - 0.01,
                value if value == "" else value + 0.01,
            ]
        rows.append(row)
    _write_rows(path, header, rows)


def _wandb_series(name: str) -> dict[str, list[tuple[int, float]]]:
    return {
        HF_REPORTED[key]: points
        for key, points in _points(name).items()
        if key in ("loss", "eval_loss", "learning_rate")
    }


def test_wandb_chart_export_holds_one_run_per_column_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "wandb_export_2026-10-01T12_00_00.000+02_00.csv"
    _chart_export(
        path,
        {
            "sweep - lr 3e-4": _wandb_series("divergence"),
            "sweep - lr 1e-4": _wandb_series("overfitting"),
        },
    )
    assert detect_format(path) == "wandb"

    kept = {"train/loss", "eval/loss", "lr"}
    first = load_run(path, run_name="sweep - lr 3e-4")
    assert first.other_runs() == ("sweep - lr 3e-4", ["sweep - lr 1e-4"])
    # The bands are not metrics, and one run's curves are not mixed with the other's.
    assert first.metric_names() == ["eval/loss", "lr", "train/loss"]
    assert _shape(run_all(first)) == [f for f in _reference("divergence") if f[2] in kept]

    second = load_run(path, run_name="sweep - lr 1e-4")
    assert second.other_runs() == ("sweep - lr 1e-4", ["sweep - lr 3e-4"])
    assert _shape(run_all(second)) == [f for f in _reference("overfitting") if f[2] in kept]
    assert load_run(path, run_name="sweep - lr 1").meta["selected_run"] == "sweep - lr 1e-4"
    # Unasked, the run with more points in the file.
    sizes = {r["name"]: r["points"] for r in first.meta["runs"]}  # type: ignore[union-attr]
    assert sizes["sweep - lr 3e-4"] != sizes["sweep - lr 1e-4"]
    default = max(sizes, key=lambda name: sizes[name])
    assert load_run(path).meta["selected_run"] == default
    other = next(name for name in sizes if name != default)

    holds = r"It holds: 'sweep - lr 3e-4', 'sweep - lr 1e-4'"
    with pytest.raises(ValueError, match=rf"--run 'sweep' is the start of 2 runs\. {holds}"):
        load_run(path, run_name="sweep")
    with pytest.raises(ValueError, match="--run 'baseline' matches no run"):
        load_run(path, run_name="baseline")

    # The reports say which run they are about.
    assert main(["analyze", str(path), "--no-color", "--run", other]) == 0
    out = capsys.readouterr().out
    assert f"run      {other}  1 of 2 in this source; --run picks another: {default}" in out
    assert main(["analyze", str(path), "--output", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["run"]["selected_run"] == default
    assert [r["name"] for r in report["run"]["runs"]] == ["sweep - lr 3e-4", "sweep - lr 1e-4"]
    html = tmp_path / "report.html"
    assert main(["analyze", str(path), "--output", "html", "--out", str(html)]) == 0
    assert f"{default} (1 of 2 in this source)" in html.read_text()


def test_wandb_chart_export_picks_the_longest_run(tmp_path: Path) -> None:
    short = {metric: points[:20] for metric, points in _wandb_series("baseline").items()}
    path = tmp_path / "wandb_export.csv"
    _chart_export(path, {"crashed-early": short, "full-run": _wandb_series("baseline")})
    run = load_run(path)
    assert run.meta["selected_run"] == "full-run"
    assert [r["points"] for r in run.meta["runs"]] == [
        sum(len(p) for p in short.values()),
        sum(len(p) for p in _wandb_series("baseline").values()),
    ]  # type: ignore[union-attr]
    assert len(run.get("train/loss")) == len(_points("baseline")["loss"])  # type: ignore[arg-type]


def test_wandb_chart_export_of_one_run_and_another_axis(tmp_path: Path) -> None:
    path = tmp_path / "wandb_export.csv"
    _chart_export(path, {"solo": _wandb_series("overfitting")}, axis="Relative Time (Process)")
    run = load_run(path)
    assert run.other_runs() is None
    assert "runs" not in run.meta
    assert run.meta["step_column"] == "(row index; the x-axis is 'Relative Time (Process)')"
    loss = run.get("train/loss")
    assert loss is not None
    assert loss.steps()[:3] == [0, 1, 2]
    assert loss.points[2].wall_time == 3.0
    # With a BOM, as a spreadsheet re-saves it.
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_run(path).metric_names() == run.metric_names()


def test_a_wandb_runs_table_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "wandb_export.csv"
    _write_rows(
        path,
        ["Name", "State", "Runtime", "learning_rate", "batch_size", "eval/loss"],
        [
            ["run-a", "finished", 5400, 4e-5, 32, 0.61],
            ["run-b", "finished", 5100, 1e-6, 32, 0.92],
            ["run-c", "crashed", 60, 1e-3, 64, ""],
        ],
    )
    with pytest.raises(ValueError, match="looks like a W&B runs table"):
        detect_format(path)
    assert main(["analyze", str(path)]) == 2
    err = capsys.readouterr().err
    assert (
        err.startswith("trainspotter: ")
        and "one row per run" in err
        and len(err.strip().splitlines()) == 1
    )
    # Forced, it is read as the CSV it is.
    assert main(["analyze", str(path), "--format", "csv", "--output", "json"]) == 0
    # A table with a step column is a log, whatever its first column is called.
    _write_rows(path, ["Name", "State", "step", "loss"], [["a", "x", 0, 1.0], ["a", "x", 1, 0.9]])
    assert detect_format(path) == "csv"


def test_run_selection_is_refused_where_there_is_one_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(EXAMPLES / "baseline.trainer_state.json"), "--run", "x"]) == 2
    assert (
        "--run picks one run out of a W&B chart export or a TensorBoard logdir"
        in capsys.readouterr().err
    )


# ------------------------------------------------------------------ TensorBoard

tb_writer = pytest.importorskip("tensorboard.summary.writer.event_file_writer")
tb_proto = pytest.importorskip("tensorboard.compat.proto.event_pb2")
tb_summary = pytest.importorskip("tensorboard.compat.proto.summary_pb2")
pytest.importorskip("tbparse")


def _events(
    logdir: Path,
    series: dict[str, list[tuple[int, float]]],
    wall0: float = 1_790_000_000.0,
    suffix: str = "",
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    writer = tb_writer.EventFileWriter(str(logdir), filename_suffix=suffix)
    for tag, points in series.items():
        for step, value in points:
            value_proto = tb_summary.Summary.Value(tag=tag, simple_value=value)
            writer.add_event(
                tb_proto.Event(
                    wall_time=wall0 + 2.0 * step,
                    step=step,
                    summary=tb_summary.Summary(value=[value_proto]),
                )
            )
    writer.close()


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_tensorboard_log_of_a_transformers_run(tmp_path: Path, name: str) -> None:
    series = {
        HF_REPORTED[key]: points for key, points in _points(name).items() if key in HF_REPORTED
    }
    _events(tmp_path / "runs" / "Oct01_12-00-00_host", series)
    run = load_run(tmp_path / "runs" / "Oct01_12-00-00_host")
    assert {"train/loss", "lr"} <= set(run.metric_names())
    assert run.other_runs() is None
    # Event files hold float32: findings are the same, on the same steps.
    assert _shape(run_all(run)) == _reference(name)
    # Pointed at the parent directory, the one run is still the run.
    assert _shape(run_all(load_run(tmp_path / "runs"))) == _reference(name)


def test_tensorboard_keras_layout_is_one_run(tmp_path: Path) -> None:
    series = _points("overfitting")
    _events(
        tmp_path / "logs" / "train",
        {"epoch_loss": series["loss"], "epoch_learning_rate": series["learning_rate"]},
    )
    _events(tmp_path / "logs" / "validation", {"epoch_loss": series["eval_loss"]})
    run = load_run(tmp_path / "logs")
    assert run.other_runs() is None
    assert run.metric_names() == ["eval/loss", "lr", "train/loss"]
    assert len(run.get("train/loss")) == len(series["loss"])  # type: ignore[arg-type]
    assert len(run.get("eval/loss")) == len(series["eval_loss"])  # type: ignore[arg-type]
    wanted = [f for f in _reference("overfitting") if f[2] in {"train/loss", "eval/loss", "lr"}]
    assert _shape(run_all(run)) == wanted
    assert any(f[0] == "overfitting" for f in wanted)


def test_tensorboard_add_scalars_layout_is_one_run(tmp_path: Path) -> None:
    """`SummaryWriter.add_scalars("Loss", {"train": ..., "val": ...})`: a directory per curve."""
    series = _points("overfitting")
    _events(tmp_path / "exp", {"lr": series["learning_rate"]})
    _events(tmp_path / "exp" / "Loss_train", {"Loss": series["loss"]})
    _events(tmp_path / "exp" / "Loss_val", {"Loss": series["eval_loss"]})
    run = load_run(tmp_path / "exp")
    assert run.other_runs() is None
    assert run.metric_names() == ["eval/loss", "lr", "train/loss"]
    assert any(f.detector == "overfitting" for f in run_all(run))


def test_tensorboard_logdir_of_several_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    diverged = {"Loss/train": _points("divergence")["loss"]}
    fine = {"Loss/train": _points("baseline")["loss"], "Loss/val": _points("baseline")["eval_loss"]}
    _events(tmp_path / "runs" / "lr1e-3", diverged)
    _events(tmp_path / "runs" / "lr3e-4" / "train", {"epoch_loss": fine["Loss/train"]})
    _events(tmp_path / "runs" / "lr3e-4" / "validation", {"epoch_loss": fine["Loss/val"]})

    run = load_run(tmp_path / "runs")
    # The run with the most points; its split directories are not runs of their own.
    assert run.other_runs() == ("lr3e-4", ["lr1e-3"])
    assert run.metric_names() == ["eval/loss", "train/loss"]
    assert not any(f.severity == "error" for f in run_all(run))

    other = load_run(tmp_path / "runs", run_name="lr1e-3")
    assert other.metric_names() == ["train/loss"]
    assert len(other.get("train/loss")) == len(diverged["Loss/train"])  # type: ignore[arg-type]
    assert any(f.detector == "divergence" for f in run_all(other))

    assert (
        main(
            ["analyze", str(tmp_path / "runs"), "--no-color", "--run", "lr1e", "--fail-on", "error"]
        )
        == 1
    )
    assert (
        "run      lr1e-3  1 of 2 in this source; --run picks another: lr3e-4"
        in capsys.readouterr().out
    )
    assert main(["analyze", str(tmp_path / "runs"), "--run", "lr"]) == 2
    assert "is the start of 2 runs" in capsys.readouterr().err


def test_tensorboard_resumed_run_keeps_what_the_restart_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loss = _points("baseline")["loss"]
    cut = loss[len(loss) * 2 // 3][0]
    back = loss[len(loss) // 2][0]
    # The first session ran to `cut` and its last checkpoint was at `back`. The values it logged
    # after that checkpoint are marked, so that reading them back would show.
    _events(
        tmp_path / "run",
        {"train/loss": [(s, v if s < back else v + 100.0) for s, v in loss if s < cut]},
    )
    _events(
        tmp_path / "run",
        {"train/loss": [(s, v) for s, v in loss if s >= back]},
        wall0=1_790_500_000.0,
        suffix=".resumed",
    )
    run = load_run(tmp_path / "run")
    series = run.get("train/loss")
    assert series is not None
    assert series.steps() == [s for s, _ in loss]
    assert max(series.values()) < 50
    dropped = sum(1 for s, _ in loss if back <= s < cut)
    assert dropped > 5
    assert run.overwritten_points() == dropped
    assert main(["analyze", str(tmp_path / "run"), "--no-color"]) == 0
    assert (
        f"resumed  {dropped} points from before a restart were written again and are left out"
        in capsys.readouterr().out
    )


# ------------------------------------------------------------------ throughput from wall time


def _timed(steps: list[int], seconds_per_step: list[float]) -> Run:
    run = Run(source_format="synthetic", source_path="<test>")
    clock = 0.0
    previous = steps[0]
    for step, rate in zip(steps, seconds_per_step, strict=True):
        clock += (step - previous) * rate
        previous = step
        run.add_point("train/loss", step, 1.0, wall_time=clock)
    run.finalize()
    return run


def test_throughput_from_wall_time_is_per_step_not_per_logged_point() -> None:
    # Logged every 10 steps, then every 50, at a steady 0.2 s per step: nothing slowed down.
    steps = [*range(0, 500, 10), *range(500, 3000, 50)]
    steady = _timed(steps, [0.2] * len(steps))
    assert [f for f in run_all(steady) if f.detector == "throughput"] == []
    # The same cadence with steps that really take three times as long by the end.
    slowing = _timed(steps, [0.2 if step < 1500 else 0.6 for step in steps])
    found = [f for f in run_all(slowing) if f.detector == "throughput"]
    assert len(found) == 1
    assert found[0].evidence["baseline_median_s"] == pytest.approx(0.2)
    assert found[0].evidence["recent_median_s"] == pytest.approx(0.6)
    assert found[0].evidence["ratio"] == pytest.approx(3.0)
