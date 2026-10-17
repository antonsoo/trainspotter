"""Readers turn a training log, in whatever format it was written, into a
:class:`trainspotter.model.Run`. Every reader is honest about what it can
and can't parse: malformed rows are counted. Rejected CSV rows with readable
coordinates and counter-free JSONL records also retain unknown observations, so
trajectory checks cannot bridge those known positions of lost data.
"""

from __future__ import annotations

import csv
from pathlib import Path

from trainspotter.model import Run

from .hf import read_hf_trainer_state
from .lightning import read_lightning_csv
from .tabular import _delimiter, read_csv, read_jsonl
from .wandb import is_chart_export, looks_like_runs_table, read_wandb_csv

__all__ = [
    "load_run",
    "read_hf_trainer_state",
    "read_csv",
    "read_jsonl",
    "read_wandb_csv",
    "read_lightning_csv",
    "read_mlflow",
]

_FORMATS = ("hf", "csv", "jsonl", "wandb", "lightning", "tensorboard", "mlflow")


def detect_format(path: str | Path) -> str:
    """Guess a log format from its path/contents. Best-effort, not magic:
    callers can always override with --format."""
    p = Path(path)
    if p.is_dir():
        from .mlflow_files import is_file_store

        if is_file_store(p):
            return "mlflow"
        return "tensorboard"
    suffix = p.suffix.lower()
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return "mlflow"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        try:
            with p.open(encoding="utf-8-sig") as fh:
                head = fh.read(4096)
            obj_start = head.lstrip()[:1]
            if obj_start == "{" and '"log_history"' in head:
                return "hf"
        except OSError:
            pass
        return "hf"
    if suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as fh:
            header = fh.readline()
            fh.seek(0)
            fieldnames = next(csv.reader(fh, delimiter=_delimiter(header)), [])
        bookkeeping = {"_step", "_runtime", "_timestamp"} & {k.strip().lower() for k in fieldnames}
        if bookkeeping or is_chart_export(fieldnames):
            return "wandb"
        if looks_like_runs_table(fieldnames):
            raise ValueError(
                f"{p}: this looks like a W&B runs table (one row per run: Name, State, Runtime, "
                "...), not the log of a run. Export a run's history, or a chart's CSV, instead; "
                "--format csv reads it as it is."
            )
        return "csv"
    from .mlflow import is_sqlite

    if is_sqlite(p):
        return "mlflow"
    raise ValueError(
        f"cannot guess format for {p!s}; pass --format explicitly "
        f"(one of {', '.join(_FORMATS)})"
    )


def load_run(
    path: str | Path, fmt: str | None = None, run_name: str | None = None,
    *, step_key: str | None = None, chart_single_run: str | None = None,
) -> Run:
    """Load a Run from `path`, auto-detecting the format unless `fmt` is given.

    `run_name` picks one run out of a source that holds several (a W&B chart export, a
    TensorBoard logdir, or a native MLflow store); see `readers.common.select_run`.
    `step_key` selects an exact integer training-step column in tabular inputs;
    the caller asserts this column measures training progress, not logger calls.
    `chart_single_run` asserts that every W&B chart value column belongs to this
    exact display name; it preserves separators within metric names.
    """
    p = Path(path)
    fmt = fmt or detect_format(p)
    if chart_single_run is not None and fmt != "wandb":
        raise ValueError("--chart-single-run applies only to W&B chart CSV exports")
    if step_key is not None and fmt not in ("csv", "jsonl", "wandb", "lightning", "mlflow"):
        raise ValueError(
            "--step-key selects a column in CSV, JSONL, W&B, Lightning or MLflow input"
        )
    if run_name is not None and fmt not in ("wandb", "tensorboard", "mlflow"):
        raise ValueError(
            f"--run picks one run out of a W&B chart export, TensorBoard logdir or MLflow store; "
            f"{p} is read as {fmt}, which holds one run"
        )
    if fmt == "hf":
        run = read_hf_trainer_state(p)
    elif fmt == "csv":
        run = read_csv(p, step_key=step_key)
    elif fmt == "jsonl":
        run = read_jsonl(p, step_key=step_key)
    elif fmt == "wandb":
        run = read_wandb_csv(p, run_name, step_key=step_key, chart_single_run=chart_single_run)
    elif fmt == "lightning":
        run = read_lightning_csv(p, step_key=step_key)
    elif fmt == "mlflow":
        run = read_mlflow(p, run_name, step_key=step_key)
    elif fmt == "tensorboard":
        from .tensorboard import read_tensorboard  # optional dependency

        run = read_tensorboard(p, run_name)
    else:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(_FORMATS)}")
    run.finalize()
    return run


def read_mlflow(
    path: str | Path, run_name: str | None = None, *, step_key: str | None = None,
) -> Run:
    from .mlflow import read_mlflow as read

    return read(path, run_name, step_key=step_key)
