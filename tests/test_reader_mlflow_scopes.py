"""Real MLflow qualified log calls must not become a training trajectory."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from trainspotter import cli
from trainspotter.detectors import run_all
from trainspotter.readers import load_run

CAPTURE = Path(__file__).resolve().parents[1] / "studies/mlflow-scopes/native"
ORACLE = json.loads((CAPTURE / "oracle.json").read_text())


def altered(tmp_path: Path, statement: str, parameters: tuple = ()) -> Path:
    database = tmp_path / "scopes.sqlite"
    shutil.copyfile(CAPTURE / "scopes.sqlite", database)
    with sqlite3.connect(database) as db:
        db.execute(statement, parameters)
    return database


def test_native_scoped_histories_retain_every_sdk_run_observation() -> None:
    database = CAPTURE / "scopes.sqlite"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == ORACLE["database"]["sha256"]
    for case in ORACLE["cases"]:
        run = load_run(database, run_name=case["run_id"], step_key="step")
        names = {raw: name for name, raw in run.meta["metric_sources"].items()}
        assert set(names) == set(case["sdk_run_histories"])
        for raw, points in case["sdk_run_histories"].items():
            assert sorted((p.step, p.value, round(p.wall_time * 1000))
                          for p in run.get(names[raw]).points) == sorted(
                (p["step"], p["value"], p["timestamp"]) for p in points)
        assert "skipped_entries" not in run.meta
        assert run.input_complete() == (case["name"] == "unqualified")
        assert run_all(run) == []
        if case["name"] == "unqualified":
            assert run.get("train/loss") is not None
            assert "metric_scopes" not in run.meta["mlflow"]
        else:
            evidence = run.meta["mlflow"]["metric_scopes"]
            assert evidence["recorded_rows"] == 20
            assert evidence["groups"] == sorted(case["sql_model_scopes"],
                                               key=lambda scope: scope["model_id"])
            assert "not additional run observations" in evidence["relationship"]
            assert "cannot recover" in " ".join(run.input_notes())
            canonical = run.get("train/loss")
            if case["name"] == "unqualified-alias":
                assert len(canonical.points) == 20
                assert run.meta["metric_sources"]["train/loss"] == "train_loss"
            else:
                assert canonical is None
            expected = 20 if case["name"] == "coalesced-coordinate" else 40
            if case["name"] == "unqualified-alias":
                expected = 20
            assert len(run.get(names["loss"]).points) == expected


@pytest.mark.parametrize("command", ["analyze", "watch"])
def test_uncertain_native_scope_cannot_pass_an_error_gate(command: str) -> None:
    args = [command, str(CAPTURE / "scopes.sqlite"), "--run", "mixed-dataset",
            "--step-key", "step", "--fail-on", "error", "--no-color"]
    if command == "watch":
        args.append("--once")
    assert cli.main(args) == 2


def test_missing_dataset_columns_are_reported_without_inventing_dataset_absence(
    tmp_path: Path,
) -> None:
    database = altered(tmp_path, "ALTER TABLE logged_model_metrics DROP COLUMN dataset_digest")
    with sqlite3.connect(database) as db:
        db.execute("ALTER TABLE logged_model_metrics DROP COLUMN dataset_name")
    run = load_run(database, run_name="mixed-dataset")
    assert not run.input_complete()
    assert run.meta["mlflow"]["metric_scopes"]["available_dataset_fields"] == []
    assert run.get("train/loss") is None


@pytest.mark.parametrize("mutation, message", [
    ("ALTER TABLE logged_model_metrics RENAME COLUMN run_id TO source_id", "schema"),
    ("UPDATE logged_model_metrics SET model_id = ''", "identity"),
    ("UPDATE logged_model_metrics SET dataset_digest = x'FFFF'", "identity"),
])
def test_ambiguous_scope_schema_or_identity_is_not_silently_ignored(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    database = altered(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        load_run(database, run_name="mixed-dataset")


def test_scope_evidence_is_bounded_without_losing_qualified_keys(tmp_path: Path) -> None:
    database = altered(tmp_path,
                       "UPDATE logged_model_metrics SET model_id = model_id || metric_step")
    run = load_run(database, run_name="mixed-dataset")
    evidence = run.meta["mlflow"]["metric_scopes"]
    assert len(evidence["groups"]) == 10
    assert evidence["omitted_groups"] == 10
    assert evidence["recorded_rows"] == 20
    assert evidence["qualified_keys"] == ["loss"]
    assert len(run.get("loss [MLflow run aggregate]").points) == 40


def test_literal_metric_cannot_be_overwritten_by_a_scope_annotation(tmp_path: Path) -> None:
    # This deliberately authored SQLite key is outside the native SDK key rules.
    database = altered(tmp_path, "UPDATE metrics SET key = 'loss [MLflow run aggregate]' "
                       "WHERE key = 'train_loss'")
    run = load_run(database, run_name="unqualified-alias")
    assert set(run.metrics) == {"loss [MLflow run aggregate]", "loss [MLflow run aggregate 2]"}
    assert all(len(series.points) == 20 for series in run.metrics.values())
