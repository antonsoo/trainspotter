from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_run

from trainspotter.detectors.grad_norm import GradNormDetector
from trainspotter.model import Run


def test_explosion_is_found_near_its_step() -> None:
    values = [1.0 + 0.05 * ((-1) ** i) for i in range(80)]
    values[50] = 40.0
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    explosions = [f for f in findings if f.title == "Gradient-norm explosion"]
    assert len(explosions) == 1
    assert explosions[0].evidence["peak_step"] == 50


def test_concentration_describes_the_norms_without_inferring_clipping() -> None:
    values = [0.5 + 0.05 * i for i in range(20)] + [1.0] * 55
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    assert len(findings) == 1
    assert findings[0].title == "Gradient norms concentrated near their maximum"
    assert findings[0].evidence["clipping_status"] == "unknown"
    assert findings[0].evidence["fraction_at_ceiling"] == 1


def test_stable_norms_have_no_false_alarm() -> None:
    values = [1.0 + 0.02 * ((-1) ** i) for i in range(80)]
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    assert findings == []


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("length", [1, 3, 50])
def test_nonfinite_norm_is_observed_without_inventing_an_infinite_ceiling(value, length) -> None:
    run = make_run(grad_norm=[1.0] * (length - 1) + [value])
    with np.errstate(all="raise"):
        findings = GradNormDetector().run(run)
    assert [f.title for f in findings] == ["Non-finite gradient norm"]
    assert findings[0].evidence["count"] == 1
    assert findings[0].step_start == length - 1


def test_nonfinite_gap_separates_gradient_levels() -> None:
    # The first reading after a missing interval cannot use earlier readings
    # as if they were a contiguous local baseline.
    run = make_run(grad_norm=[1.0] * 20 + [math.nan] + [100.0] * 20)
    findings = GradNormDetector().run(run)
    assert [f.title for f in findings] == ["Non-finite gradient norm"]


def test_nonfinite_norm_survives_an_ambiguous_step_without_a_trajectory() -> None:
    run = Run()
    for step, value in ((0, 1.0), (1, 2.0), (1, math.inf), (2, math.nan)):
        run.add_point("grad_norm", step, value)
    run.finalize()
    findings = GradNormDetector().run(run)
    assert len(findings) == 1
    assert findings[0].evidence["count"] == 2
    assert findings[0].evidence["distinct_bad_steps"] == 2
    assert findings[0].evidence["trajectory_status"] == "conflicting_values"
    assert not run.input_complete()


def test_agreeing_repeated_nonfinite_norms_count_once() -> None:
    run = Run()
    for _ in range(3):
        run.add_point("grad_norm", 10, math.nan)
    run.finalize()
    assert GradNormDetector().run(run)[0].evidence["count"] == 1
