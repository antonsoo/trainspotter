"""Seeded fuzz of the whole path from a log file to a report.

Whatever a log looks like (no rows, one row, repeated or unordered steps, NaN and
infinite values, rows of the wrong shape), `analyze` either writes a report or
exits with a one-line message. The JSON report must parse under a strict parser
and the HTML report must contain no coordinate that is not a number.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pytest

from trainspotter.cli import main

_ODD_KINDS = ["diverge", "constant", "zero", "noisy", "huge", "negative", "broken", "spiky"]
_KINDS = ["decay"] * 6 + _ODD_KINDS
_METRICS = [
    "loss", "train_loss", "eval_loss", "val_loss", "learning_rate", "lr", "grad_norm",
    "eval_accuracy", "samples_per_second", "step_time", "epoch", "weird metric", "", "loss/train",
]  # fmt: skip
_BAD_RECORDS: list[Any] = [
    {},
    {"step": None},
    {"step": "abc", "loss": "x"},
    {"loss": [1, 2]},
    {"step": 3, "loss": None},
    {"step": 3, "loss": True},
    {"step": 3, "loss": {"a": 1}},
    {"step": 1e400, "loss": 1.0},
    {"step": float("nan"), "loss": 1.0},
]


def _value(rng: random.Random, kind: str, i: int, n: int) -> float:
    if kind == "decay":
        return 3.0 * math.exp(-i / max(1, n / 3)) + 0.1 + rng.gauss(0, 0.02)
    if kind == "diverge":
        return 2.0 * math.exp(-i / n) if i < n * 0.7 else 2.0 + (i - n * 0.7) ** 1.5
    if kind == "constant":
        return 1.0
    if kind == "zero":
        return 0.0
    if kind == "noisy":
        return rng.random() * 10
    if kind == "huge":
        return rng.choice([1e308, 1e300, -1e308, 1e-320, 5e15])
    if kind == "negative":
        return -abs(rng.gauss(0, 3))
    if kind == "broken":
        return rng.choice([float("nan"), float("inf"), -float("inf"), 1.0, 2.0])
    return 1.0 + (50.0 if rng.random() < 0.05 else 0.0) + rng.gauss(0, 0.01)  # spiky


def _records(rng: random.Random) -> list[Any]:
    n = rng.choice([0, 1, 2, 3, 5, 10, 40, 200, 600])
    kinds = {m: rng.choice(_KINDS) for m in rng.sample(_METRICS, rng.randint(1, 6))}
    odd_steps = ["repeated", "unordered", "large", "negative", "half", "fixed", "gap"]
    steps = rng.choice(["rising"] * 3 + odd_steps)
    records: list[Any] = []
    for i in range(n):
        step: float = {
            "rising": i * 10,
            "repeated": i // 2,
            "unordered": rng.randint(0, n),
            "large": i * 10**12,
            "negative": i - n,
            "half": i * 0.5,
            "fixed": 7,
            "gap": i * 10 + (10**6 if i > n // 2 else 0),
        }[steps]
        record: Any = {"step": step}
        for metric, kind in kinds.items():
            if rng.random() >= 0.15:
                record[metric] = _value(rng, kind, i, max(n, 1))
        records.append(rng.choice(_BAD_RECORDS) if rng.random() < 0.02 else record)
    return records


def _cell(rng: random.Random, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return rng.choice([str(value), "NaN", ""])
    return str(value).replace(",", ";")


def _write(rng: random.Random, records: list[Any], directory: Path) -> tuple[Path, str]:
    fmt = rng.choice(["csv", "jsonl", "hf", "wandb", "lightning"])
    if fmt in ("csv", "wandb", "lightning"):
        columns: list[str] = []
        for record in records:
            columns += [key for key in record if key not in columns]
        header = columns
        if fmt == "wandb":
            header = ["_step" if c == "step" else c for c in columns] + ["_runtime"]
        lines = [",".join(header)]
        for record in records:
            cells = [_cell(rng, record.get(c)) for c in columns]
            if rng.random() < 0.01:
                cells = cells[: rng.randint(0, len(cells))]
            if rng.random() < 0.01:
                cells += ["stray", "cells"]
            lines.append(",".join(cells))
        text = "" if rng.random() < 0.03 else "\n".join(lines) + "\n"
        path = directory / "log.csv"
    elif fmt == "jsonl":
        lines = []
        for record in records:
            lines.append(json.dumps(record))
            if rng.random() < 0.01:
                lines.append(rng.choice(["", "{", "[1,2]", "null", "5", '"x"']))
        text = "\n".join(lines)
        path = directory / "log.jsonl"
    else:
        history: Any = records
        if rng.random() < 0.05:
            history = rng.choice([None, 5, "x", {}, [1, 2, None]])
        state: Any = {"log_history": history, "global_step": rng.choice([None, 10, "x"])}
        if rng.random() < 0.03:
            state = rng.choice([[], 5, None, {"log_history": {"a": 1}}, {}])
        text = json.dumps(state)
        path = directory / "trainer_state.json"
    path.write_text(text, encoding="utf-8")
    return path, fmt


def _refuse_constant(name: str) -> float:
    raise AssertionError(f"{name} in the JSON report")


_NUMBER_ATTRIBUTE = re.compile(r'\s(?:x|y|x1|x2|y1|y2|width|height)="([^"]*)"')


@pytest.mark.filterwarnings("error")  # a numpy warning on stderr is part of the output too
@pytest.mark.parametrize("block", range(6))
def test_any_log_gives_a_report_or_a_one_line_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block: int
) -> None:
    reports = 0
    for seed in range(block * 100, (block + 1) * 100):
        rng = random.Random(seed)
        path, fmt = _write(rng, _records(rng), tmp_path)
        output = rng.choice(["terminal", "json", "html"])
        argv = ["analyze", str(path), "--output", output, "--no-color"]
        if fmt in ("wandb", "lightning") or rng.random() < 0.2:
            argv += ["--format", fmt]
        if output == "html":
            argv += ["--out", str(tmp_path / "report.html")]
        try:
            code = main(argv)
        except Exception as exc:  # the failure this test exists to catch
            raise AssertionError(f"seed {seed}: {argv} raised {type(exc).__name__}: {exc}") from exc
        captured = capsys.readouterr()
        if code == 2:
            message = captured.err.strip()
            one_line = message.startswith("trainspotter: ") and "\n" not in message
            assert one_line, f"seed {seed}: {message}"
            continue
        assert code == 0, f"seed {seed}: exit {code}"
        reports += 1
        if output == "json":
            report = json.loads(captured.out, parse_constant=_refuse_constant)
            assert report["run"]["skipped_rows"] >= 0
        elif output == "html":
            html = (tmp_path / "report.html").read_text(encoding="utf-8")
            for attribute in _NUMBER_ATTRIBUTE.findall(html):
                assert math.isfinite(float(attribute)), f"seed {seed}: coordinate {attribute!r}"
            for drawn in re.findall(r'<path[^>]* d="([^"]*)"', html):
                assert "nan" not in drawn and "inf" not in drawn, f"seed {seed}: {drawn[:60]}"
    assert reports >= 60, "most generated logs are rejected: the detectors and reports go untested"
