# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas==3.0.6"]
# ///
"""Retain pandas-written CSVs using the committed divergence log's measurements.

The experiment names and chart bands are controlled layouts, not W&B downloads.
Bands repeat each center value solely to identify the chart layout. No new
training, hosted W&B account, or uncertainty measurement is involved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
METRICS = {"loss": "train/loss", "eval_loss": "eval/loss", "learning_rate": "lr"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def observations(rows, key):
    result = []
    for row in rows:
        if row[key] == "":
            continue
        value = float(row[key])
        result.append([int(row["step"]), value.hex() if math.isfinite(value) else str(value)])
    return {"count": len(result), "sha256": digest(json.dumps(result).encode())}


def main():
    fixtures = STUDY / "fixtures"
    fixtures.mkdir(exist_ok=True)
    source = ROOT / "examples/divergence.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []

    def export(name, columns, data, delimiter, **metadata):
        path = fixtures / (name + ".csv")
        # Object strings retain the source's literal numeric spellings and
        # sparse cells. pandas supplies CSV quoting and record framing.
        pd.DataFrame(data, columns=columns, dtype=object).to_csv(
            path, index=False, sep=delimiter, lineterminator="\r\n"
        )
        with path.open(newline="") as stream:
            actual = list(csv.reader(stream, delimiter=delimiter, strict=True))
        assert actual == [columns, *data]
        records.append(
            {
                "name": name,
                "file": "fixtures/" + path.name,
                "delimiter": delimiter,
                "sha256": digest(path.read_bytes()),
                **metadata,
            }
        )

    history = {canonical: observations(rows, key) for key, canonical in METRICS.items()}
    export(
        "control",
        ["global_step", *METRICS],
        [[row["step"], *[row[key] for key in METRICS]] for row in rows],
        ",",
        kind="generic",
        expected=history,
    )
    layouts = [
        ("chart-plain", ",", "baseline"),
        ("chart-semicolons", ",", "seed=0; lr=0.001; wd=0.01"),
        ("chart-tabs", ",", "seed=0\tlr=0.001\twd=0.01"),
        ("chart-quotes", ",", 'seed=0, note="main"; lr=0.001; wd=0.01'),
        ("chart-multiline", ",", "seed=0; lr=0.001;\nwd=0.01"),
        ("semicolon-writer", ";", "seed=0, lr=0.001, wd=0.01"),
        ("tab-writer", "\t", "seed=0; lr=0.001; wd=0.01"),
    ]
    for name, delimiter, label in layouts:
        columns = [
            "global_step",
            *[f"{label} - {key}{suffix}" for key in METRICS for suffix in ("", "__MIN", "__MAX")],
        ]
        data = [[row["step"], *[row[key] for key in METRICS for _ in range(3)]] for row in rows]
        export(name, columns, data, delimiter, kind="chart", label=label, expected=history)
    export(
        "history-punctuation",
        ["_step", "global_step", *METRICS, "note;meta;seed"],
        [
            [str(index), row["step"], *[row[key] for key in METRICS], "recorded"]
            for index, row in enumerate(rows)
        ],
        ",",
        kind="history",
        expected=history,
    )
    export(
        "generic-punctuation",
        ["global_step", *METRICS, "note;meta;seed"],
        [[row["step"], *[row[key] for key in METRICS], "recorded"] for row in rows],
        ",",
        kind="generic",
        expected=history,
    )
    receipt = {
        "scope": __doc__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "measurementSource": {
            "file": "examples/divergence.csv",
            "sha256": digest(source.read_bytes()),
            "rows": len(rows),
        },
        "cases": records,
    }
    (STUDY / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"exports": len(records), "sourceRows": len(rows), "pandas": pd.__version__}))


if __name__ == "__main__":
    main()
