"""Readers turn a training log, in whatever format it was written, into a
:class:`trainspotter.model.Run`. Every reader is honest about what it can
and can't parse: malformed rows are skipped with a count, not silently
dropped or fatally raised.
"""

from __future__ import annotations

from pathlib import Path

from trainspotter.model import Run

from .hf import read_hf_trainer_state
from .lightning import read_lightning_csv
from .tabular import read_csv, read_jsonl
from .wandb import read_wandb_csv

__all__ = [
    "load_run",
    "read_hf_trainer_state",
    "read_csv",
    "read_jsonl",
    "read_wandb_csv",
    "read_lightning_csv",
]

_FORMATS = ("hf", "csv", "jsonl", "wandb", "lightning", "tensorboard")


def detect_format(path: str | Path) -> str:
    """Guess a log format from its path/contents. Best-effort, not magic:
    callers can always override with --format."""
    p = Path(path)
    if p.is_dir():
        return "tensorboard"
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        try:
            with p.open(encoding="utf-8") as fh:
                head = fh.read(4096)
            obj_start = head.lstrip()[:1]
            if obj_start == "{" and '"log_history"' in head:
                return "hf"
        except OSError:
            pass
        return "hf"
    if suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as fh:
            header = fh.readline()
        if header.startswith("_step") or ",_step" in header or "_runtime" in header:
            return "wandb"
        return "csv"
    raise ValueError(
        f"cannot guess format for {p!s}; pass --format explicitly "
        f"(one of {', '.join(_FORMATS)})"
    )


def load_run(path: str | Path, fmt: str | None = None) -> Run:
    """Load a Run from `path`, auto-detecting the format unless `fmt` is given."""
    p = Path(path)
    fmt = fmt or detect_format(p)
    if fmt == "hf":
        run = read_hf_trainer_state(p)
    elif fmt == "csv":
        run = read_csv(p)
    elif fmt == "jsonl":
        run = read_jsonl(p)
    elif fmt == "wandb":
        run = read_wandb_csv(p)
    elif fmt == "lightning":
        run = read_lightning_csv(p)
    elif fmt == "tensorboard":
        from .tensorboard import read_tensorboard  # optional dependency

        run = read_tensorboard(p)
    else:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(_FORMATS)}")
    run.finalize()
    return run
