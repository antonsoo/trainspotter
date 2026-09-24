"""Reader tests against real files this test writes to disk -- not mocks,
not remembered sample data. Each format's on-disk shape is exactly what its
docstring in `trainspotter/readers/*.py` claims to read."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trainspotter.readers import detect_format, load_run
from trainspotter.readers.hf import read_hf_trainer_state
from trainspotter.readers.lightning import read_lightning_csv
from trainspotter.readers.tabular import read_csv, read_jsonl
from trainspotter.readers.wandb import read_wandb_csv


def test_hf_trainer_state(tmp_path: Path) -> None:
    log_history = [
        {"loss": 2.5, "learning_rate": 1e-5, "epoch": 0.1, "step": 10, "grad_norm": 1.1},
        {"eval_loss": 2.3, "eval_accuracy": 0.4, "epoch": 0.1, "step": 10},
        {"loss": 2.1, "learning_rate": 2e-5, "epoch": 0.2, "step": 20, "grad_norm": 1.3},
    ]
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": log_history, "best_metric": 2.3}))

    run = read_hf_trainer_state(path)
    run.finalize()

    assert run.source_format == "hf"
    assert run.get("train/loss") is not None
    assert run.get("train/loss").steps() == [10, 20]  # type: ignore[union-attr]
    assert run.get("eval/loss") is not None
    assert run.get("eval/loss").steps() == [10]  # type: ignore[union-attr]
    assert run.get("lr") is not None
    assert run.get("grad_norm") is not None
    assert run.get("eval/accuracy") is not None
    assert detect_format(path) == "hf"


def test_hf_trainer_state_skips_malformed_entries(tmp_path: Path) -> None:
    log_history = [{"loss": 1.0, "step": 1}, {"loss": 2.0}, "not-a-dict"]
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": log_history}))

    run = read_hf_trainer_state(path)

    assert run.meta["skipped_entries"] == 2
    assert run.get("train/loss").steps() == [1]  # type: ignore[union-attr]


def test_hf_trainer_state_excludes_the_final_run_summary_entry(tmp_path: Path) -> None:
    # Trainer.train() appends one more entry after the real per-step logs:
    # a run-level summary reusing the final step number, with an *average*
    # train_loss -- not a per-step reading. It must not land in train/loss.
    log_history = [
        {"loss": 2.0, "learning_rate": 1e-4, "step": 10},
        {"loss": 1.5, "learning_rate": 1e-4, "step": 20},
        {
            "step": 20,
            "train_runtime": 12.3,
            "train_samples_per_second": 40.0,
            "train_steps_per_second": 5.0,
            "train_loss": 1.75,
            "total_flos": 123.0,
        },
    ]
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": log_history}))

    run = read_hf_trainer_state(path)
    run.finalize()

    series = run.get("train/loss")
    assert series is not None
    assert series.values() == [2.0, 1.5]  # the 1.75 average must not appear here
    assert run.meta["train_summary"]["train_runtime"] == 12.3


def test_generic_csv(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "learning_rate"])
        for i in range(5):
            writer.writerow([i, 2.0 - 0.1 * i, 1e-4])

    run = read_csv(path)
    run.finalize()

    assert run.get("train/loss").steps() == [0, 1, 2, 3, 4]  # type: ignore[union-attr]
    assert run.meta["step_column"] == "step"
    assert detect_format(path) == "csv"


def test_generic_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    with path.open("w") as fh:
        for i in range(5):
            fh.write(json.dumps({"step": i, "loss": 2.0 - 0.1 * i}) + "\n")
        fh.write("not json\n")  # malformed line, should be skipped not fatal

    run = read_jsonl(path)

    assert run.get("train/loss") is not None
    assert len(run.get("train/loss")) == 5  # type: ignore[arg-type]
    assert run.meta["skipped_lines"] == 1


def test_wandb_history_csv(tmp_path: Path) -> None:
    path = tmp_path / "wandb_history.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["_step", "_runtime", "train/loss", "eval/loss"])
        writer.writerow([0, 1.2, 2.0, ""])
        writer.writerow([1, 2.5, 1.8, 1.9])

    run = read_wandb_csv(path)
    run.finalize()

    assert run.get("train/loss").steps() == [0, 1]  # type: ignore[union-attr]
    assert run.get("eval/loss").steps() == [1]  # type: ignore[union-attr]
    assert run.get("train/loss").points[0].wall_time == 1.2  # type: ignore[union-attr]
    assert detect_format(path) == "wandb"


def test_wandb_csv_without_step_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="_step"):
        read_wandb_csv(path)


def test_lightning_csv_logger(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "epoch", "train_loss", "val_loss"])
        writer.writerow([0, 0, 2.0, ""])
        writer.writerow([1, 0, 1.9, ""])
        writer.writerow([1, 0, "", 2.1])  # Lightning logs train/val on separate rows

    run = read_lightning_csv(path)
    run.finalize()

    assert run.source_format == "lightning"
    assert run.get("train/loss").steps() == [0, 1]  # type: ignore[union-attr]
    assert run.get("eval/loss").steps() == [1]  # type: ignore[union-attr]


def test_load_run_auto_detects_format(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps({"step": 0, "loss": 1.0}) + "\n")
    run = load_run(path)
    assert run.source_format == "jsonl"


def test_reads_the_committed_real_transformers_trainer_state() -> None:
    # examples/transformers_tinygpt2.trainer_state.json came from a real
    # transformers.Trainer run (see generate_transformers_example.py), not
    # a hand-written fixture. This pins that it keeps parsing cleanly, and
    # specifically that the run's average-loss summary entry (see hf.py's
    # docstring) never leaks into the per-step train/loss series.
    path = Path(__file__).parent.parent / "examples" / "transformers_tinygpt2.trainer_state.json"
    run = read_hf_trainer_state(path)
    run.finalize()

    series = run.get("train/loss")
    assert series is not None
    assert len(series) > 100
    summary_avg = run.meta["train_summary"]["train_loss"]
    assert summary_avg not in series.values()[-3:]  # the average, not a per-step tail value
