from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import make_run

from trainspotter.detectors.loss_floor import LossFloorDetector
from trainspotter.model import Run
from trainspotter.readers import load_run

CAPTURES = Path(__file__).resolve().parents[1] / "examples/diagnostic-evidence"


def test_implausibly_fast_drop_is_flagged() -> None:
    values = [2.3] + [0.001] * 4 + [2.3 * 0.97**i for i in range(5, 100)]
    run = make_run(**{"train/loss": values})

    findings = LossFloorDetector().run(run)

    assert len(findings) == 1
    assert findings[0].severity == "info"


def test_normal_gradual_descent_is_not_flagged() -> None:
    values = [2.3 * 0.98**i + 0.1 for i in range(100)]
    run = make_run(**{"train/loss": values})

    findings = LossFloorDetector().run(run)

    assert findings == []


def test_same_recorded_training_run_is_not_made_early_by_sparse_logging() -> None:
    detector = LossFloorDetector()
    assert detector.run(load_run(CAPTURES / "slow-dense.csv")) == []
    assert detector.run(load_run(CAPTURES / "slow-sparse.csv")) == []
    finding, = detector.run(load_run(CAPTURES / "fast-dense.csv"))
    assert finding.step_end == 18
    assert finding.evidence["fraction_of_logged_span"] == .018


@pytest.mark.parametrize("start", [0, 1_000_000, 2**53 - 1001])
@pytest.mark.parametrize("first_low", [100, 101, 900])
def test_early_boundary_depends_on_logged_step_span_not_absolute_step_or_row(start, first_low):
    run = Run()
    for offset, value in [(0, 2), (first_low, .001), (910, .001), (950, .001), (1000, .001)]:
        run.add_point("train/loss", start + offset, value)
    run.finalize()
    findings = LossFloorDetector().run(run)
    assert bool(findings) == (first_low == 100)
    if findings:
        assert findings[0].evidence["first_logged_step"] == start
        assert findings[0].evidence["last_logged_step"] == start + 1000
        assert findings[0].step_end == start + 100


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
@pytest.mark.parametrize("position", [0, 2, 8])
def test_invalid_or_negative_early_reading_prevents_a_relative_drop_claim(value, position):
    values = [2.3] + [.001] * 99
    values[position] = value
    assert LossFloorDetector().run(make_run(**{"train/loss": values})) == []


@pytest.mark.parametrize("axis", ["logging", "row"])
def test_nontraining_axis_cannot_establish_an_early_drop(axis):
    run = make_run(**{"train/loss": [2.3] + [.001] * 99})
    if axis == "logging":
        run.meta["step_axis"] = "logging_step"
    else:
        run.meta["step_column"] = "(row index)"
    assert LossFloorDetector().run(run) == []
