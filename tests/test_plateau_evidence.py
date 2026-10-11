"""A weak linear slope alone does not establish a stable loss level."""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest
from conftest import make_run

from trainspotter.detectors.plateau import PlateauDetector, _window_extrema
from trainspotter.model import Run


def test_recovered_rise_and_later_improvement_cannot_form_one_flat_interval() -> None:
    values = [1.0] * 50 + [4.0] * 20 + [1.0] * 50 + [0.1] * 40
    findings = PlateauDetector().run(make_run(**{"train/loss": values}))
    assert findings
    assert all(f.step_end < 50 or f.step_start >= 70 for f in findings)
    assert all(f.step_end < 120 for f in findings)  # the later floor is convergence


@pytest.mark.parametrize("gap", [math.nan, math.inf, -math.inf])
def test_a_gap_cannot_disable_convergence_in_the_following_segment(gap: float) -> None:
    values = [1.0] * 30 + [gap] * 10
    values += [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40
    findings = PlateauDetector().run(make_run(**{"train/loss": values}))
    assert [(f.step_start, f.step_end) for f in findings] == [(0, 29)]
    assert findings[0].evidence["segment_start_step"] == 0
    assert findings[0].evidence["segment_end_step"] == 29


def test_a_complete_minimum_window_can_establish_a_constant_level() -> None:
    findings = PlateauDetector().run(make_run(**{"train/loss": [5.0] * 30}))
    assert [(f.step_start, f.step_end) for f in findings] == [(0, 29)]


def test_symmetric_large_excursion_does_not_become_a_flat_window() -> None:
    # Endpoints and OLS slope are equal, but the middle is four times higher.
    values = [1.0] * 20 + [4.0] * 20 + [1.0] * 20
    detector = PlateauDetector(min_window=30, convergence_frac=2)
    assert detector.run(make_run(**{"train/loss": values})) == []


def test_an_lr_reading_in_the_future_cannot_suppress_an_earlier_stall() -> None:
    run = Run(source_format="synthetic", source_path="future-lr")
    for step in range(100):
        run.add_point("train/loss", step, 5.0)
    run.add_point("lr", 80, 0.05)
    run.add_point("lr", 90, 1.0)
    run.finalize()
    assert PlateauDetector().run(run)


@pytest.mark.parametrize("gap", [math.nan, math.inf, -1.0])
def test_lr_decay_suppression_cannot_bridge_invalid_schedule_evidence(gap: float) -> None:
    run = Run(source_format="synthetic", source_path="lr-gap")
    for step in range(50, 150):
        run.add_point("train/loss", step, 5.0)
    for step, value in [(0, 1.0), (20, gap), (40, 0.05)]:
        run.add_point("lr", step, value)
    run.finalize()
    assert PlateauDetector().run(run)


@pytest.mark.parametrize("window", [1, 2, 15, 30, 201, 400])
def test_extrema_match_literal_windows_without_mutating_input(window: int) -> None:
    values = np.random.default_rng(817).integers(-100, 100, 800).astype(float)[::2]
    values.flags.writeable = False
    minimum, maximum = _window_extrema(values, window)
    assert minimum.tolist() == [min(values[i:i + window])
                                for i in range(len(values) - window + 1)]
    assert maximum.tolist() == [max(values[i:i + window])
                                for i in range(len(values) - window + 1)]


def test_reported_level_range_has_an_independent_sorted_window_witness() -> None:
    values = [5 * 0.9**i + 5 for i in range(40)]
    values += [5 + .001 * (-1)**i + (.5 if 28 <= i < 33 else 0) for i in range(60)]
    values += [5 * .9**i + .5 for i in range(40)]
    findings = PlateauDetector(min_window=20).run(make_run(**{"train/loss": values}))
    assert len(findings) == 1  # short noise remains one plateau
    for finding in findings:
        half = int(finding.evidence["smoothing_window"]) // 2
        # Use the standard library's sorted median, independently of the
        # vectorized rolling implementation and its extrema queues.
        smooth = [statistics.median(values[max(0, i - half):i + half + 1])
                  for i in range(len(values))]
        region = smooth[finding.step_start:finding.step_end + 1]
        low, high = min(region), max(region)
        assert finding.evidence["level_min"] == low
        assert finding.evidence["level_max"] == high
        assert (high - low) / max(abs(low), abs(high), 1e-12) < .02
        assert "raw readings range" in finding.message
