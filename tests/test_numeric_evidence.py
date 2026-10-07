"""Known boundary signals must not create or erase diagnostic evidence."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_run

from trainspotter.detectors.base import ols_slope
from trainspotter.detectors.divergence import DivergenceDetector
from trainspotter.detectors.plateau import PlateauDetector
from trainspotter.detectors.throughput import ThroughputDetector


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_loss_needs_one_observation_not_three(value: float) -> None:
    run = make_run(**{"train/loss": [value]})
    findings = DivergenceDetector().run(run)
    assert len(findings) == 1
    assert findings[0].evidence["count"] == 1
    assert findings[0].severity == "error"


def test_exactly_constant_data_has_no_evidence_of_a_nonzero_slope() -> None:
    slope, se, p = ols_slope(np.arange(30, dtype=float), np.ones(30))
    assert (slope, se, p) == (0, 0, 1)
    findings = PlateauDetector().run(make_run(**{"train/loss": [5.0] * 100}))
    assert len(findings) == 1
    assert (findings[0].step_start, findings[0].step_end) == (0, 99)


def test_missing_readings_cannot_become_a_plateau() -> None:
    values = [5.0] * 5 + [math.nan] * 20 + [5.0] * 10
    assert PlateauDetector().run(make_run(**{"train/loss": values})) == []


@pytest.mark.parametrize("metric", ["step_time", "throughput"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_invalid_timing_windows_do_not_produce_a_numeric_comparison(metric, bad) -> None:
    values = [0.1] * 20 + [bad] * 20
    with np.errstate(all="raise"):
        assert ThroughputDetector().run(make_run(**{metric: values})) == []


def test_throughput_units_are_not_mislabeled_as_seconds_per_step() -> None:
    run = make_run(**{"throughput": [1000.0] * 20 + [250.0] * 20})
    findings = ThroughputDetector().run(run)
    assert len(findings) == 1
    assert findings[0].evidence["ratio"] == 4.0
    assert findings[0].evidence["unit"] == "s/unit"
    assert "step time" not in findings[0].message


def test_clock_reversal_does_not_shift_the_recent_window_into_older_history() -> None:
    run = make_run(**{"train/loss": [1.0] * 30})
    from trainspotter.model import Point

    run.metrics["train/loss"].points = [
        Point(i, 1.0, float(i if i < 20 else 40 - i)) for i in range(30)
    ]
    assert ThroughputDetector().run(run) == []


def test_large_finite_timings_do_not_overflow_the_median() -> None:
    run = make_run(**{"step_time": [1e308] * 20 + [1.6e308] * 20})
    with np.errstate(all="raise"):
        findings = ThroughputDetector().run(run)
    assert len(findings) == 1
    assert findings[0].evidence["ratio"] == pytest.approx(1.6)
    assert findings[0].evidence["recent_median_s"] == 1.6e308
