"""Native local MLflow metric histories, without starting or importing MLflow.

SQLite is opened read-only in one transaction. Read `metrics`, never the
latest-value cache. A native MLflow step is a caller-supplied coordinate, not
necessarily an optimizer update; --step-key step explicitly asserts that unit.
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trainspotter.model import Run

from .common import metric_names, select_run, try_step

_METRIC_COLUMNS = {"key", "value", "timestamp", "step", "is_nan", "run_uuid"}


@dataclass(frozen=True)
class _RunInfo:
    run_id: str
    name: str
    experiment_id: str
    status: str
    lifecycle_stage: str


def is_sqlite(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(16) == b"SQLite format 3\x00"


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    record = db.execute("SELECT type FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    if record is None or record[0] != "table":
        raise ValueError(f"not a supported MLflow SQLite store: missing {table!r} table")
    # Table names come only from this module's fixed schema, not database content.
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def _select(
    run: Run, infos: dict[str, _RunInfo], counts: dict[str, int], wanted: str | None,
    store: str,
) -> str:
    if not infos:
        raise ValueError("MLflow store contains no runs")
    candidates = counts
    if wanted is None:
        candidates = {key: count for key, count in counts.items()
                      if infos[key].lifecycle_stage == "active"}
        if not candidates:
            raise ValueError("MLflow store has no active runs; select a recorded run ID with --run")
    elif wanted not in infos and not any(key.startswith(wanted) for key in infos):
        matches = [key for key, info in infos.items() if info.name == wanted]
        if len(matches) > 1:
            raise ValueError(
                f"MLflow display name {wanted!r} belongs to {len(matches)} runs; "
                f"select an exact run ID or unique ID prefix: {', '.join(matches)}"
            )
        if matches:
            wanted = matches[0]
    chosen = select_run(run, candidates, wanted, "MLflow store")
    inventory = [{**asdict(info), "observations": counts[key],
                  "history_read": key == chosen} for key, info in infos.items()]
    if len(infos) > 1:
        run.meta["runs"] = [
            {"name": key, "points": counts[key], "display_name": info.name,
             "lifecycle_stage": info.lifecycle_stage} for key, info in infos.items()
        ]
    run.meta["mlflow"] = {"store": store, "run": asdict(infos[chosen]), "runs": inventory,
                          "history": "full recorded metric histories", "read_only": True,
                          "issues": [], "omitted_issues": 0}
    return chosen


def _issue(
    run: Run, metric: str, reason: str, step: int | None = None, *, count_row: bool = True,
) -> None:
    count = run.meta.get("skipped_entries", 0)
    if count_row:
        run.meta["skipped_entries"] = int(count) + 1 if isinstance(count, int) else 1
    evidence = run.meta["mlflow"]
    assert isinstance(evidence, dict)
    issues = evidence["issues"]
    if len(issues) < 10:
        issues.append({"metric": metric, "reason": reason, "step": step})
    else:
        evidence["omitted_issues"] += 1


def _observation(
    run: Run, metric: str, step_raw: Any, value: Any, timestamp: Any, is_nan: Any = 0,
) -> None:
    step = try_step(step_raw) if type(step_raw) is int else None
    if step is None:
        _issue(run, metric, "unreadable or unsupported integer step")
        return
    millis = try_step(timestamp) if type(timestamp) is int else None
    wall_time = millis / 1000 if millis is not None else None
    if wall_time is not None and round(wall_time * 1000) != millis:
        wall_time = None
    if wall_time is None:
        _issue(run, metric, "unreadable or inexact integer millisecond timestamp", step)
    if type(value) not in (int, float) or type(is_nan) is not int or is_nan not in (0, 1):
        _issue(run, metric, "unreadable metric value or NaN flag", step,
               count_row=wall_time is not None)
        run.add_unreadable(metric, step, wall_time, known_metric=True)
        return
    run.add_point(metric, step, math.nan if is_nan else float(value), wall_time)


def _scoped_metric_keys(db: sqlite3.Connection, run: Run, chosen: str) -> set[str]:
    """Model-qualified log calls can also populate the unqualified run table.

    Never subtract matching rows: SQL can coalesce an unqualified call and a
    qualified call with identical key/value/step/time. Scope cannot then be
    recovered from the aggregate. Retain its values under an explicit name.
    """
    record = db.execute(
        "SELECT type FROM sqlite_master WHERE name = 'logged_model_metrics'"
    ).fetchone()
    if record is None:
        return set()  # Older MLflow stores have no logged-model metric signal.
    if record[0] != "table":
        raise ValueError("unsupported MLflow logged-model metrics object; expected a table")
    columns = _table_columns(db, "logged_model_metrics")
    if not {"metric_name", "model_id", "run_id"} <= columns:
        raise ValueError(
            "unsupported MLflow logged-model metric schema; cannot establish run metric scopes"
        )
    # These expressions are fixed identifiers, not database-supplied SQL.
    dataset = "dataset_name" if "dataset_name" in columns else "NULL"
    digest = "dataset_digest" if "dataset_digest" in columns else "NULL"
    keys: set[str] = set()
    groups: list[dict[str, str | int | None]] = []
    group_count = qualified_rows = 0
    for metric, model, data_name, data_digest, count in db.execute(
        f"SELECT metric_name, model_id, {dataset}, {digest}, count(*) "
        "FROM logged_model_metrics WHERE run_id = ? "
        f"GROUP BY metric_name, model_id, {dataset}, {digest} "
        f"ORDER BY metric_name, model_id, {dataset}, {digest}", (chosen,)
    ):
        if (not isinstance(metric, str) or not metric
                or not isinstance(model, str) or not model
                or any(value is not None and not isinstance(value, str)
                       for value in (data_name, data_digest))):
            raise ValueError("unreadable MLflow logged-model metric identity or dataset scope")
        keys.add(metric)
        group_count += 1
        qualified_rows += count
        if len(groups) < 10:
            groups.append({"metric": metric, "model_id": model, "dataset_name": data_name,
                           "dataset_digest": data_digest, "recorded_rows": count})
    if keys:
        evidence = run.meta["mlflow"]
        assert isinstance(evidence, dict)
        evidence["metric_scopes"] = {
            "qualified_keys": sorted(keys), "recorded_rows": qualified_rows,
            "groups": groups, "omitted_groups": group_count - len(groups),
            "available_dataset_fields": sorted(columns & {"dataset_name", "dataset_digest"}),
            "relationship": (
                "Logged-model rows may also occur in the run aggregate; these are not "
                "additional run observations. Model/dataset histories are not reconstructed."
            ),
        }
        run.meta["ambiguous_metric_sources"] = sorted(keys)
        for metric in sorted(keys):
            _issue(run, metric, "linked model/dataset records leave the run aggregate's "
                   "metric scope unresolved", count_row=False)
    return keys


def _metric_names(run: Run, raw_names: list[str], scoped: set[str]) -> dict[str, str]:
    if not scoped:
        return metric_names(run, raw_names)
    # Unqualified keys keep their normal alias ranking. A scoped `loss` must
    # not claim train/loss ahead of a separately recorded unqualified train_loss.
    names = metric_names(run, [raw for raw in raw_names if raw not in scoped])
    used = set(names.values()) | set(raw_names)
    for raw in raw_names:
        if raw in scoped:
            name = f"{raw} [MLflow run aggregate]"
            suffix = 2
            while name in used:
                name = f"{raw} [MLflow run aggregate {suffix}]"
                suffix += 1
            names[raw] = name
            used.add(name)
    names = {raw: names[raw] for raw in raw_names}
    run.meta["metric_sources"] = {name: raw for raw, name in names.items()}
    evidence = run.meta["mlflow"]
    assert isinstance(evidence, dict)
    evidence["metric_scopes"]["retained_aggregates"] = {
        raw: names[raw] for raw in raw_names if raw in scoped
    }
    return names


def _sqlite(path: Path, wanted: str | None, step_key: str | None) -> Run:
    run = Run(source_format="mlflow", source_path=str(path))
    run.meta["step_column"] = "step"
    run.meta["step_axis"] = "step" if step_key else "logging_step"
    if step_key:
        run.meta["step_selection"] = "explicit"
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        with closing(connection) as db:
            db.execute("PRAGMA query_only = ON")
            db.execute("PRAGMA trusted_schema = OFF")
            db.execute("BEGIN")
            if _table_columns(db, "metrics") != _METRIC_COLUMNS:
                raise ValueError(
                    "unsupported MLflow metric schema; expected key, value, timestamp, step, "
                    "is_nan and run_uuid. Export separate per-run CSV histories instead."
                )
            columns = _table_columns(db, "runs")
            if not {"run_uuid", "experiment_id", "status", "lifecycle_stage"} <= columns:
                raise ValueError("not a supported MLflow SQLite store: incomplete runs schema")
            name = "name" if "name" in columns else "NULL"
            infos: dict[str, _RunInfo] = {}
            for key, label, experiment, status, lifecycle in db.execute(
                f"SELECT run_uuid, {name}, experiment_id, status, lifecycle_stage "
                "FROM runs ORDER BY run_uuid"
            ):
                if not isinstance(key, str) or not key or key in infos:
                    raise ValueError("MLflow run IDs must be unique nonempty strings")
                if lifecycle not in ("active", "deleted"):
                    raise ValueError(f"MLflow run {key!r} has an unsupported lifecycle stage")
                infos[key] = _RunInfo(key, label if isinstance(label, str) else key,
                                      str(experiment), str(status), lifecycle)
            counts = dict.fromkeys(infos, 0)
            count_query = "SELECT run_uuid, count(*) FROM metrics GROUP BY run_uuid"
            for key, count in db.execute(count_query):
                if key not in infos:
                    raise ValueError(
                        "MLflow metrics contain a history with no matching run identity"
                    )
                counts[key] = count
            chosen = _select(run, infos, counts, wanted, "sqlite")
            scoped = _scoped_metric_keys(db, run, chosen)
            raw_names = [row[0] for row in db.execute(
                "SELECT DISTINCT key FROM metrics WHERE run_uuid = ? ORDER BY key", (chosen,)
            ) if isinstance(row[0], str) and row[0]]
            names = _metric_names(run, raw_names, scoped)
            for raw, value, timestamp, step, nan_flag in db.execute(
                "SELECT key, value, timestamp, step, is_nan FROM metrics "
                "WHERE run_uuid = ? ORDER BY key, step, timestamp, value", (chosen,)
            ):
                if not isinstance(raw, str) or raw not in names:
                    _issue(run, "(unknown metric)", "metric key is not a nonempty string")
                    continue
                _observation(run, names[raw], step, value, timestamp, nan_flag)
    except sqlite3.Error as exc:
        raise ValueError(f"could not read a consistent MLflow SQLite snapshot: {exc}") from exc
    return run


def read_mlflow(
    path: str | Path, run_name: str | None = None, *, step_key: str | None = None,
) -> Run:
    """Read one native local run, selected by ID/prefix or unique display name."""
    if step_key not in (None, "step"):
        raise ValueError(
            "native MLflow has one step coordinate; "
            "use --step-key step to assert training updates"
        )
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"MLflow source does not exist: {source}")
    if source.is_dir():
        from .mlflow_files import read_file_store

        return read_file_store(source, run_name, step_key)
    if not is_sqlite(source):
        raise ValueError("expected a native MLflow SQLite database file")
    return _sqlite(source, run_name, step_key)
