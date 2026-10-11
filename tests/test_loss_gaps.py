"""A nonfinite observation must not silently splice two finite loss histories."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_run

from trainspotter.detectors.base import levels
from trainspotter.detectors.divergence import DivergenceDetector
from trainspotter.detectors.setbacks import SetbackDetector
from trainspotter.detectors.spikes import SpikeDetector


def _run(values: list[float]):
    return make_run(**{"train/loss": values})


@pytest.mark.parametrize("gap", [math.nan, math.inf, -math.inf])
def test_level_change_across_gap_is_not_a_continuous_rise(gap: float) -> None:
    findings = DivergenceDetector().run(_run([1.0] * 50 + [gap] * 10 + [3.0] * 50))
    assert [f.title for f in findings] == ["Non-finite training loss"]
    assert findings[0].evidence["count"] == 10


def test_nonfinite_tail_cannot_describe_an_earlier_level_as_the_final_loss() -> None:
    values = [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [math.nan] * 20
    findings = DivergenceDetector().run(_run(values))
    assert [f.title for f in findings] == ["Non-finite training loss"]
    assert findings[0].step_end == 99


def test_no_duration_or_recovery_is_inferred_across_missing_levels() -> None:
    values = [5.0] * 20 + [1.0] * 40 + [math.nan] * 10 + [4.0] * 20
    values += [math.nan] * 10 + [1.0] * 40
    assert SetbackDetector().run(_run(values)) == []


def test_first_value_after_gap_has_no_local_spike_baseline() -> None:
    values = [5.0] * 20 + [1.0] * 40 + [math.nan] * 10 + [4.0] + [1.0] * 40
    assert SpikeDetector().run(_run(values)) == []


def test_real_recovered_excursion_after_gap_is_assessed_on_its_own_evidence() -> None:
    values = [1.0] * 30 + [math.nan] * 10
    values += [5.0] * 20 + [1.0] * 40 + [4.0] * 20 + [1.0] * 40
    findings = SetbackDetector().run(_run(values))
    assert len(findings) == 1
    finding = findings[0]
    assert finding.title == "Setback"
    assert (finding.step_start, finding.step_end) == (100, 120)
    assert finding.evidence["steps_lost"] == 20
    assert finding.evidence["segment_start_step"] == 40
    assert finding.evidence["segment_end_step"] == 159
    assert finding.evidence["share_of_segment"] == pytest.approx(20 / 119)
    assert "finite segment" in finding.message
    assert "share_of_run" not in finding.evidence


def test_real_spikes_on_both_sides_of_gap_remain_separate() -> None:
    section = [5.0] * 20 + [1.0] * 30 + [9.0] + [1.0] * 30
    values = section + [math.nan] + section
    findings = SpikeDetector().run(_run(values))
    assert [(f.title, f.step_start, f.step_end) for f in findings] == [
        ("Loss spike", 50, 50), ("Loss spike", 132, 132),
    ]
    assert all("finite segment" in f.message for f in findings)


def test_final_segment_can_still_establish_rising_loss() -> None:
    values = [1.0] * 40 + [math.nan] * 10 + [3.0 + i * 0.07 for i in range(50)]
    findings = DivergenceDetector().run(_run(values))
    trajectory = [f for f in findings if "non-finite" not in f.title.lower()]
    assert len(trajectory) == 1
    assert (trajectory[0].step_start, trajectory[0].step_end) == (50, 99)
    assert trajectory[0].evidence["start_level"] > 3
    assert "final finite segment" in trajectory[0].message


@pytest.mark.parametrize("values", [[math.nan] * 20, [1.0] * 5 + [math.nan] + [5.0] * 5])
def test_short_or_absent_finite_segments_do_not_acquire_a_combined_level(values) -> None:
    assert levels(np.array(values)) is None
    assert [f.title for f in DivergenceDetector().run(_run(values))] == [
        "Non-finite training loss",
    ]
