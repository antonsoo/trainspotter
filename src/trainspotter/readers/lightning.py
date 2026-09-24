"""PyTorch Lightning `CSVLogger` reader.

`CSVLogger` writes `<save_dir>/<name>/version_N/metrics.csv` with a `step`
column, an `epoch` column, and one column per logged metric. Lightning logs
different metrics on different rows (a training-step row has `train_loss`
filled and `val_loss` blank, a validation-epoch row is the reverse), so most
cells are empty; blanks are skipped rather than treated as zero.
"""

from __future__ import annotations

from pathlib import Path

from trainspotter.model import Run

from .tabular import read_csv


def read_lightning_csv(path: str | Path) -> Run:
    run = read_csv(path)
    run.source_format = "lightning"
    return run
