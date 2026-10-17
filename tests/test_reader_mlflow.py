"""Native SDK histories, scoped selection, live records and storage preservation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import sqlite3
from pathlib import Path

import pytest

from trainspotter import cli
from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.timing import assess_timing

CAPTURE = Path(__file__).resolve().parents[1] / "studies/mlflow-native/fixtures/mlflow-3.17.1"
ORACLE = json.loads(gzip.decompress((CAPTURE / "histories.json.gz").read_bytes()))


def store_path(kind: str, root: Path = CAPTURE) -> Path:
    return root / ("mlflow.db" if kind == "sqlite" else "mlruns")


def copied(tmp_path: Path) -> Path:
    destination = tmp_path / "capture"
    shutil.copytree(CAPTURE, destination)
    return destination


def sql(root: Path, query: str, parameters: tuple = ()) -> None:
    with sqlite3.connect(store_path("sqlite", root)) as db:
        db.execute(query, parameters)


def native_run(kind: str, name: str) -> dict:
    return next(
        run
        for store in ORACLE["stores"]
        if store["kind"] == kind
        for run in store["runs"]
        if run["name"] == name
    )


def run_folder(root: Path, name: str) -> Path:
    info = native_run("files", name)
    return root / "mlruns" / info["experimentId"] / info["runId"]


def value_token(value: float) -> str:
    return str(value) if not math.isfinite(value) else value.hex()


@pytest.mark.parametrize("version", ["2.17.2", "3.17.1"])
@pytest.mark.parametrize("kind", ["sqlite", "files"])
def test_every_record_matches_native_sdk_full_history(kind: str, version: str) -> None:
    root = CAPTURE.parent / f"mlflow-{version}"
    oracle = json.loads(gzip.decompress((root / "histories.json.gz").read_bytes()))
    for expected in next(store for store in oracle["stores"] if store["kind"] == kind)["runs"]:
        run = load_run(store_path(kind, root), run_name=expected["runId"])
        sources = {raw: canonical for canonical, raw in run.meta["metric_sources"].items()}
        assert set(sources) == set(expected["histories"])
        for key, history in expected["histories"].items():
            observed = [
                (point.step, value_token(point.value), round(point.wall_time * 1000))
                for point in run.get(sources[key]).points
            ]
            native = [
                (point["step"], value_token(float(point["value"])), point["timestamp"])
                for point in history
            ]
            assert sorted(observed) == sorted(native), (kind, expected["name"], key)
        assert run.meta["selected_run"] == expected["runId"]
        assert run.source_format == "mlflow"
        assert run.axis_kind() == "logging_step"
        assert run.meta["mlflow"]["run"]["name"] == expected["name"]


@pytest.mark.parametrize("version", ["2.17.2", "3.17.1"])
def test_native_files_remain_byte_identical(version: str) -> None:
    root = CAPTURE.parent / f"mlflow-{version}"
    manifest = json.loads((root / "capture.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("kind", ["sqlite", "files"])
def test_real_training_replays_keep_diagnosis_and_separate_runs(kind: str) -> None:
    healthy = load_run(store_path(kind), run_name="baseline", step_key="step")
    assert healthy.input_complete()
    assert run_all(healthy) == []
    overfit = load_run(store_path(kind), run_name="overfitting", step_key="step")
    assert overfit.input_complete()
    assert any(f.title == "Possible overfitting" for f in run_all(overfit))
    divergent = load_run(store_path(kind), run_name="divergence", step_key="step")
    assert divergent.input_complete()
    if kind == "files":
        assert any(f.detector == "divergence" and f.severity == "error" for f in run_all(divergent))
    else:
        # Native SQLite replaces Inf with a finite extreme; do not invent its old value.
        assert math.isfinite(divergent.get("train/loss").points[-1].value)
        assert any(f.title == "Loss spike" for f in run_all(divergent))
        assert "cannot reconstruct infinities" in " ".join(divergent.input_notes())
    assert len(healthy.meta["mlflow"]["runs"]) == 8
    assert sum(info["history_read"] for info in healthy.meta["mlflow"]["runs"]) == 1


@pytest.mark.parametrize("kind", ["sqlite", "files"])
def test_native_steps_do_not_silently_measure_optimizer_updates(kind: str) -> None:
    default = load_run(store_path(kind), run_name="slow-training")
    assert assess_timing(default).status == "unavailable"
    assert "MLflow logged steps" in assess_timing(default).reason
    asserted = load_run(store_path(kind), run_name="slow-training", step_key="step")
    timing = assess_timing(asserted)
    assert timing.status == "compared"
    assert timing.ratio > 2
    assert any(f.detector == "throughput" for f in run_all(asserted))
    stable = load_run(store_path(kind), run_name="stable-training", step_key="step")
    assert not any(f.detector == "throughput" for f in run_all(stable))
    assert default.axis_kind() == "logging_step"
    assert asserted.axis_kind() == "step"


@pytest.mark.parametrize("kind", ["sqlite", "files"])
def test_run_id_prefix_and_unique_name_choose_the_same_history(kind: str) -> None:
    info = native_run(kind, "baseline")
    named = load_run(store_path(kind), run_name="baseline")
    prefix = load_run(store_path(kind), run_name=info["runId"][:10])
    assert prefix.metrics == named.metrics
    assert prefix.meta["selected_run"] == info["runId"]


def test_duplicate_display_names_never_combine_native_runs(tmp_path: Path) -> None:
    root = copied(tmp_path)
    sql(root, "UPDATE runs SET name = 'same-name'")
    with pytest.raises(ValueError, match="belongs to 8 runs"):
        load_run(store_path("sqlite", root), run_name="same-name")
    expected = native_run("sqlite", "baseline")
    selected = load_run(store_path("sqlite", root), run_name=expected["runId"])
    assert selected.meta["selected_run"] == expected["runId"]


def test_deleted_runs_require_explicit_selection(tmp_path: Path) -> None:
    root = copied(tmp_path)
    sql(root, "UPDATE runs SET lifecycle_stage = 'deleted'")
    with pytest.raises(ValueError, match="no active runs"):
        load_run(store_path("sqlite", root))
    result = load_run(store_path("sqlite", root), run_name="baseline")
    assert result.meta["mlflow"]["run"]["lifecycle_stage"] == "deleted"
    assert result.input_complete()


@pytest.mark.parametrize("kind", ["sqlite", "files"])
@pytest.mark.parametrize("name", ["conflicting-steps", "default-step"])
def test_conflicting_native_readings_cannot_pass_as_one_trajectory(kind: str, name: str) -> None:
    run = load_run(store_path(kind), run_name=name)
    assert not run.input_complete()
    assert run.analysis_view().get("train/loss") is None
    assert sum(len(group.readings) for group in run.repeated_steps()["train/loss"]) >= 2
    assert (
        cli.main(
            ["analyze", str(store_path(kind)), "--run", name, "--fail-on", "error", "--no-color"]
        )
        == 2
    )


def test_unknown_values_stay_unknown_and_do_not_contaminate_sibling_runs(tmp_path: Path) -> None:
    root = copied(tmp_path)
    bad = native_run("sqlite", "overfitting")["runId"]
    sql(
        root,
        "UPDATE metrics SET value = 'not-a-number' "
        "WHERE run_uuid = ? AND key = 'loss' AND step = 7",
        (bad,),
    )
    damaged = load_run(store_path("sqlite", root), run_name=bad)
    assert not damaged.input_complete()
    assert next(p for p in damaged.get("train/loss").points if p.step == 7).value is None
    assert load_run(store_path("sqlite", root), run_name="baseline").input_complete()


def test_unterminated_file_record_waits_for_its_terminator(tmp_path: Path) -> None:
    root = copied(tmp_path)
    metric = run_folder(root, "baseline") / "metrics/loss"
    with metric.open("ab") as stream:
        stream.write(b"1700001000000 0.1234 400")
    partial = load_run(store_path("files", root), run_name="baseline")
    assert not partial.input_complete()
    assert partial.get("train/loss").points[-1].step == 399
    assert "unterminated" in partial.meta["mlflow"]["issues"][0]["reason"]
    with metric.open("ab") as stream:
        stream.write(b"\n")
    complete = load_run(store_path("files", root), run_name="baseline")
    assert complete.input_complete()
    assert complete.get("train/loss").points[-1].step == 400
    assert complete.get("train/loss").points[-1].value == 0.1234


def test_bad_file_records_keep_their_metric_coordinate_without_a_fake_nan(tmp_path: Path) -> None:
    root = copied(tmp_path)
    metric = run_folder(root, "baseline") / "metrics/loss"
    with metric.open("ab") as stream:
        stream.write(b"1700001000000 unreadable 400\n1700001001000 0.1 401\n")
    run = load_run(store_path("files", root), run_name="baseline")
    assert not run.input_complete()
    assert run.get("train/loss").points[-2].value is None
    assert run.get("train/loss").points[-1].value == 0.1
    assert not any(f.title == "Non-finite training loss" for f in run_all(run))
    assert load_run(store_path("files", root), run_name="overfitting").input_complete()


@pytest.mark.parametrize(
    "metadata", [b"run_id: first\nrun_id: second\n", b"\xff", b"[not, a, mapping]\n"]
)
def test_unreadable_metadata_is_an_input_error(tmp_path: Path, metadata: bytes) -> None:
    root = copied(tmp_path)
    folder = run_folder(root, "baseline")
    (folder / "meta.yaml").write_bytes(metadata)
    with pytest.raises(ValueError, match="metadata"):
        load_run(folder, fmt="mlflow")


def test_duplicate_copies_of_a_run_need_a_single_source_directory(tmp_path: Path) -> None:
    root = copied(tmp_path)
    folder = run_folder(root, "baseline")
    shutil.copytree(folder, folder.parent / "copied-run")
    with pytest.raises(ValueError, match="multiple directories"):
        load_run(store_path("files", root))
    assert load_run(folder).input_complete()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sqlite_sidecars_cannot_be_report_destinations(tmp_path: Path, suffix: str) -> None:
    root = copied(tmp_path)
    database = store_path("sqlite", root)
    target = Path(str(database) + suffix)
    # This also protects a future sidecar; no invalid fake WAL is installed.
    assert (
        cli.main(
            [
                "analyze",
                str(database),
                "--run",
                "baseline",
                "--output",
                "json",
                "--out",
                str(target),
            ]
        )
        == 2
    )
    assert not target.exists()


@pytest.mark.parametrize(
    "relative", ["meta.yaml", "metrics/loss", "metrics/new-report", "tags/mlflow.runName"]
)
def test_file_store_input_paths_are_not_report_scratch_space(tmp_path: Path, relative: str) -> None:
    root = copied(tmp_path)
    folder = run_folder(root, "baseline")
    target = folder / relative
    before = target.read_bytes() if target.exists() else None
    assert (
        cli.main(
            ["analyze", str(store_path("files", root)), "--run", "baseline", "--out", str(target)]
        )
        == 2
    )
    assert (target.read_bytes() if target.exists() else None) == before


def test_file_store_artifacts_can_hold_a_report(tmp_path: Path) -> None:
    root = copied(tmp_path)
    folder = run_folder(root, "baseline")
    target = folder / "artifacts/diagnosis.html"
    # A git archive retains files, not the SDK's initially empty artifacts directory.
    target.parent.mkdir(exist_ok=True)
    assert (
        cli.main(
            [
                "analyze",
                str(store_path("files", root)),
                "--run",
                "baseline",
                "--output",
                "html",
                "--out",
                str(target),
            ]
        )
        == 0
    )
    assert "MLflow run" in target.read_text()
    assert load_run(store_path("files", root), run_name="baseline").input_complete()


def test_other_sqlite_databases_are_refused_without_creating_or_modifying_files(
    tmp_path: Path,
) -> None:
    database = tmp_path / "other.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE unrelated (value REAL)")
    before = database.read_bytes()
    assert cli.main(["analyze", str(database)]) == 2
    assert database.read_bytes() == before
    absent = tmp_path / "missing.db"
    assert cli.main(["analyze", str(absent)]) == 2
    assert not absent.exists()


def test_sqlite_wal_snapshot_sees_committed_history_and_preserves_the_writer(
    tmp_path: Path,
) -> None:
    root = copied(tmp_path)
    database = store_path("sqlite", root)
    identity = native_run("sqlite", "baseline")["runId"]
    with sqlite3.connect(database) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
            ("loss", 0.123, 1_700_000_400_000, identity, 400, 0),
        )
        writer.commit()
        wal = Path(str(database) + "-wal")
        before = (database.read_bytes(), wal.read_bytes())
        committed = load_run(database, run_name=identity)
        assert committed.get("train/loss").points[-1].step == 400
        assert committed.get("train/loss").points[-1].value == 0.123
        writer.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
            ("loss", 0.124, 1_700_000_401_000, identity, 401, 0),
        )
        uncommitted = load_run(database, run_name=identity)
        assert uncommitted.get("train/loss").points[-1].step == 400
        assert (database.read_bytes(), wal.read_bytes()) == before
        writer.commit()
        assert load_run(database, run_name=identity).get("train/loss").points[-1].step == 401


def test_new_metric_identity_columns_are_not_silently_merged(tmp_path: Path) -> None:
    root = copied(tmp_path)
    sql(root, "ALTER TABLE metrics ADD COLUMN dataset_digest TEXT")
    with pytest.raises(ValueError, match="unsupported MLflow metric schema"):
        load_run(store_path("sqlite", root))


def test_dataset_qualified_file_records_stay_unknown(tmp_path: Path) -> None:
    root = copied(tmp_path)
    path = run_folder(root, "baseline") / "metrics/loss"
    with path.open("a") as handle:
        handle.write("1700000400000 900.0 400 held-out digest\n")
        handle.write("1700000401000 0.15 401\n")
    run = load_run(store_path("files", root), run_name="baseline")
    assert not run.input_complete()
    assert run.get("train/loss").points[-2].value is None
    assert run.get("train/loss").points[-2].step == 400
    assert run.get("train/loss").points[-1].value == 0.15
    assert "dataset-qualified" in run.meta["mlflow"]["issues"][0]["reason"]


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_file_store_aliases_cannot_overwrite_other_runs(tmp_path: Path, alias: str) -> None:
    root = copied(tmp_path)
    protected = run_folder(root, "overfitting") / "metrics/loss"
    target = tmp_path / "report.html"
    if alias == "symlink":
        target.symlink_to(protected)
    else:
        target.hardlink_to(protected)
    before = protected.read_bytes()
    assert (
        cli.main(
            ["analyze", str(store_path("files", root)), "--run", "baseline", "--out", str(target)]
        )
        == 2
    )
    assert protected.read_bytes() == before


def test_inexact_millisecond_clock_does_not_invent_timestamp_precision(tmp_path: Path) -> None:
    root = copied(tmp_path)
    identity = native_run("sqlite", "baseline")["runId"]
    timestamp = 2**53 - 1
    assert round((timestamp / 1000) * 1000) != timestamp
    sql(
        root,
        "UPDATE metrics SET timestamp = ? WHERE run_uuid = ? AND step = 17",
        (timestamp, identity),
    )
    run = load_run(store_path("sqlite", root), run_name=identity)
    point = next(p for p in run.get("train/loss").points if p.step == 17)
    assert point.wall_time is None
    assert point.value is not None
    assert not run.input_complete()
    assert "inexact" in run.meta["mlflow"]["issues"][0]["reason"]


def test_file_store_deleted_experiments_remain_explicitly_selectable(tmp_path: Path) -> None:
    root = copied(tmp_path)
    info = native_run("files", "baseline")
    experiment = root / "mlruns" / info["experimentId"]
    for metadata in experiment.glob("*/meta.yaml"):
        metadata.write_text(metadata.read_text().replace("lifecycle_stage: active",
                                                        "lifecycle_stage: deleted"))
    destination = root / "mlruns/.trash" / info["experimentId"]
    destination.parent.mkdir(exist_ok=True)
    experiment.rename(destination)
    with pytest.raises(ValueError, match="no active runs"):
        load_run(store_path("files", root))
    explicit = load_run(store_path("files", root), run_name=info["runId"])
    assert explicit.meta["mlflow"]["run"]["lifecycle_stage"] == "deleted"
    assert explicit.input_complete()
