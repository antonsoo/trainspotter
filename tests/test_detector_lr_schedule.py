from __future__ import annotations

import math

import pytest
from conftest import make_run, make_run_steps

from trainspotter.detectors.lr_schedule import LRScheduleDetector


def test_missing_warmup_is_flagged() -> None:
    peak = 1e-4
    values = [peak] * 10 + [peak * (0.95**i) for i in range(1, 30)]  # no ramp, straight to peak
    run = make_run(lr=values)

    findings = LRScheduleDetector().run(run)

    titles = [f.title for f in findings]
    assert "No LR warmup detected" in titles


def test_warmup_then_cosine_decay_has_no_false_alarm() -> None:
    import math

    warmup_steps = 10
    total = 100
    peak = 1e-4
    values = []
    for i in range(total):
        if i < warmup_steps:
            values.append(peak * (i + 1) / warmup_steps)
        else:
            progress = (i - warmup_steps) / (total - warmup_steps)
            values.append(peak * 0.5 * (1 + math.cos(math.pi * progress)))
    run = make_run(lr=values)

    findings = LRScheduleDetector().run(run)

    assert findings == []


def test_discontinuity_flagged_on_large_relative_jump() -> None:
    warmup = [1e-4 * (i + 1) / 10 for i in range(10)]
    decay = [1e-4 * 0.9**i for i in range(1, 20)]
    values = warmup + decay
    values[15] = 1e-6  # inject an out-of-schedule jump, an order of magnitude below its neighbors
    run = make_run(lr=values)

    findings = LRScheduleDetector().run(run)

    assert any(f.title == "LR discontinuity" for f in findings)


def test_short_warmup_logged_every_few_steps_is_not_a_discontinuity() -> None:
    # The real Hugging Face run in examples/: 20 warmup steps logged every 5 steps, so each logged
    # interval covers a quarter of the LR range - a steady ramp, not a jump.
    from conftest import make_run_steps

    steps = list(range(5, 730, 5))
    peak = 5e-4
    values = [min(peak, peak * s / 20) if s <= 20 else peak * (1 - (s - 20) / 710) for s in steps]
    findings = LRScheduleDetector().run(make_run_steps("lr", steps, values))
    assert [f.title for f in findings] == []


def test_the_real_transformers_run_has_no_lr_finding() -> None:
    from pathlib import Path

    from trainspotter.readers import load_run

    run = load_run(
        Path(__file__).parent.parent / "examples" / "transformers_tinygpt2.trainer_state.json"
    )
    assert LRScheduleDetector().run(run) == []


def test_a_jump_over_a_logging_interval_says_so() -> None:
    from conftest import make_run_steps

    steps = list(range(0, 200, 10))
    values = [1e-4] * 10 + [1e-6] * 10  # one step-decay drop between logged points 10 steps apart
    findings = LRScheduleDetector().run(make_run_steps("lr", steps, values))
    jump = next(f for f in findings if f.title == "LR discontinuity")
    assert "across one logged interval (10 steps)" in jump.message


def rebounds(values, steps=None, **options):
    run = make_run_steps("lr", steps, values) if steps is not None else make_run(lr=values)
    return [f for f in LRScheduleDetector(**options).run(run) if f.title == "LR rose after decay"]


@pytest.mark.parametrize("end", [0.07, 0.10, 0.12])
def test_smooth_rise_below_at_or_above_original_peak_is_visible(end: float) -> None:
    decay = [0.10 - i * 0.01 for i in range(10)]
    growth = [0.01 + i * (end - 0.01) / 10 for i in range(1, 11)]
    values = decay + growth
    steps = [i * 100 for i in range(len(values))]
    findings = rebounds(values, steps)
    assert len(findings) == 1
    f = findings[0]
    assert (f.step_start, f.step_end, f.severity) == (900, 1900, "info")
    assert f.evidence["earlier_peak"] == 0.1
    assert f.evidence["peak_step"] == 0
    assert f.evidence["low_value"] == pytest.approx(0.01)
    assert f.evidence["high_value"] == pytest.approx(end)
    assert f.evidence["confirmed_step"] == 1000
    assert "log alone does not establish a scheduler fault" in f.message


def test_three_observations_can_establish_a_reversal() -> None:
    assert len(rebounds([0.1, 0.01, 0.05])) == 1


def test_zero_low_is_usable_without_dividing_by_zero() -> None:
    f = rebounds([0.1, 0.05, 0.0, 0.01, 0.02])[0]
    assert f.evidence["low_value"] == 0
    assert f.evidence["rise_fraction_of_peak"] == pytest.approx(0.2)


def test_one_long_rise_is_one_episode_and_later_peaks_do_not_erase_it() -> None:
    values = [0.1, 0.08, 0.01, 0.02, 0.03, 0.04, 0.08, 0.16]
    for length in range(4, len(values) + 1):
        (f,) = rebounds(values[:length], peak_grace_frac=0)
        assert (f.step_start, f.step_end) == (2, length - 1)
        assert f.evidence["confirmed_step"] == 3
    later = rebounds(values + [0.001, 0.5], peak_grace_frac=0)
    assert [(f.step_start, f.step_end) for f in later] == [(2, 7), (8, 9)]


def test_plateaus_and_small_wiggles_do_not_split_a_rise() -> None:
    values = [0.1, 0.0995, 0.1, 0.05, 0.02, 0.02, 0.0205, 0.04, 0.0399, 0.04, 0.06]
    (f,) = rebounds(values)
    assert (f.step_start, f.step_end) == (5, 10)
    assert f.evidence["confirmed_step"] == 7
    assert rebounds([0.1, 0.0995, 0.1, 0.0999, 0.1]) == []


def test_smooth_cumulative_rise_counts_but_subthreshold_bounce_does_not() -> None:
    assert rebounds([0.1, 0.05, 0.0502, 0.0504, 0.0506, 0.0508]) == []
    (f,) = rebounds([0.1, 0.05, 0.0502, 0.0504, 0.0506, 0.0508, 0.0512])
    assert f.evidence["confirmed_step"] == 6


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [0.0, 0.05, 0.1, 0.1, 0.08, 0.01, 0.0],
        [0.1, 0.1, 0.05, 0.05, 0.01, 0.01],
        [0.05] * 20,
    ],
)
def test_warmup_decay_steps_and_constant_rates_have_no_rebound(values) -> None:
    assert rebounds(values) == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01])
def test_unusable_values_break_the_evidence_chain(bad: float) -> None:
    assert rebounds([0.1, 0.01, bad, 0.02, 0.03]) == []
    findings = rebounds([0.1, 0.01, 0.02, bad, 0.1, 0.01, 0.03])
    assert [(f.step_start, f.step_end) for f in findings] == [(1, 2), (5, 6)]


@pytest.mark.parametrize("scale", [1e-290, 1e-8, 1.0, 1e300])
def test_threshold_is_relative_to_rate_scale(scale: float) -> None:
    (f,) = rebounds([v * scale for v in [0.1, 0.05, 0.01, 0.02, 0.03]])
    assert (f.step_start, f.step_end) == (2, 4)
    assert f.evidence["rise_fraction_of_peak"] == pytest.approx(0.2)


def test_grace_is_an_explicit_point_count_option() -> None:
    values = [0.1, 0.01, 0.03] + [0.03] * 97
    assert rebounds(values, peak_grace_frac=0.05) == []
    assert len(rebounds(values)) == 1


def test_conflicting_observations_cannot_invent_a_schedule() -> None:
    run = make_run(lr=[0.1, 0.05, 0.01, 0.02, 0.03])
    run.add_point("lr", 2, 0.04)
    assert LRScheduleDetector().run(run) == []


def test_abrupt_restarts_are_not_reported_twice() -> None:
    values = [0.1 * (1 - (i % 40) / 40) for i in range(120)]
    findings = LRScheduleDetector().run(make_run(lr=values))
    assert [(f.title, f.step_start, f.step_end) for f in findings] == [
        ("No LR warmup detected", 0, 0),
        ("LR rose after decay", 39, 40),
        ("LR rose after decay", 79, 80),
    ]


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, -1.0])
def test_unusable_reading_cannot_erase_distant_real_drops_or_create_a_jump(bad):
    values = [0.1] * 20 + [0.05] * 20 + [0.025] * 20
    values[10] = bad
    findings = LRScheduleDetector().run(make_run(lr=values))
    assert [(f.title, f.step_start, f.step_end) for f in findings] == [
        ("No LR warmup detected", 0, 0),
        ("LR discontinuity", 19, 20),
        ("LR discontinuity", 39, 40),
    ]
    for finding in findings[1:]:
        assert finding.evidence["usable_lr_range"] == pytest.approx(0.075)
        assert finding.evidence["segment_start_step"] == 11
        assert finding.evidence["segment_end_step"] == 59
        assert finding.evidence["unusable_lr_readings"] == 1


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, -1.0])
def test_unknown_transition_is_not_replaced_by_the_two_nearest_usable_readings(bad):
    values = [0.1] * 20 + [bad] + [0.01] * 20
    findings = LRScheduleDetector().run(make_run(lr=values))
    assert not any(f.title == "LR discontinuity" for f in findings)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, -1.0])
def test_a_gap_resets_the_initial_rise_phase_without_inventing_a_restart(bad):
    values = [0.1] * 20 + [0.05] * 20 + [bad] + [0.001] * 10 + [0.1] * 20
    findings = LRScheduleDetector().run(make_run(lr=values))
    jumps = [f for f in findings if f.title == "LR discontinuity"]
    assert [(f.step_start, f.step_end) for f in jumps] == [(19, 20)]
    assert not any(f.title == "LR rose after decay" for f in findings)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, -1.0])
def test_unusable_beginning_is_not_evidence_of_missing_warmup(bad):
    findings = LRScheduleDetector().run(make_run(lr=[bad] + [0.1] * 40))
    assert findings == []


def test_small_segment_range_does_not_inflate_a_change_relative_to_the_whole_run():
    values = [1.0] * 20 + [math.nan] + [0.02] * 10 + [0.01] * 10
    findings = LRScheduleDetector().run(make_run(lr=values))
    assert not any(f.title == "LR discontinuity" for f in findings)


def test_discontinuity_requires_six_contiguous_usable_observations():
    values = [0.1] * 20 + [math.nan] + [0.1, 0.1, 0.01, 0.01, 0.01]
    findings = LRScheduleDetector().run(make_run(lr=values))
    assert not any(f.title == "LR discontinuity" for f in findings)


def test_unusable_readings_do_not_supply_missing_warmup_sample_count():
    values = [0.1] * 19 + [math.nan] * 20
    assert LRScheduleDetector().run(make_run(lr=values)) == []


@pytest.mark.parametrize("values", [[math.nan] * 20, [math.inf] * 20, [-1.0] * 20, [None] * 20])
def test_no_usable_lr_history_has_no_schedule_findings(values):
    assert LRScheduleDetector().run(make_run(lr=values)) == []
