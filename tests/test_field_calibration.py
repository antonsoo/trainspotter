"""The shapes real `Trainer` logs take, and what each detector should say about them.

Every curve here is synthetic, built to match a pattern from the census of
public logs in `studies/hub-census/`: the per-step scatter of a mini-batch
loss, a warmup logged every few steps, a level that climbs for a tenth of
the run and comes back. The earlier detectors were tuned on smooth generated
examples and raised a warning on three quarters of the public runs that
ended at their best loss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_run

from trainspotter.detectors import run_all
from trainspotter.detectors.base import excursions, levels
from trainspotter.detectors.divergence import DivergenceDetector
from trainspotter.detectors.grad_norm import GradNormDetector
from trainspotter.detectors.lr_schedule import LRScheduleDetector
from trainspotter.detectors.overfitting import OverfittingDetector
from trainspotter.detectors.setbacks import SetbackDetector
from trainspotter.detectors.spikes import SpikeDetector


def noisy_decay(
    n: int, seed: int, start: float = 2.0, floor: float = 0.2, sigma: float = 0.25
) -> np.ndarray:
    """An ordinary fine-tuning loss logged every step: an exponential approach to a
    floor, times log-normal scatter (heavy upper tail, as mini-batch losses have)."""
    rng = np.random.default_rng(seed)
    level = floor + (start - floor) * np.exp(-np.arange(n) / (n / 5))
    return level * rng.lognormal(0.0, sigma, n)


def with_bump(values: np.ndarray, start: int, length: int, factor: float) -> np.ndarray:
    out = values.copy()
    out[start : start + length] *= factor
    return out


# ---------------------------------------------------------------------------
# Healthy runs stay quiet.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [200, 2000, 8000])
def test_a_healthy_noisy_run_raises_nothing_above_info(n: int) -> None:
    alarms = 0
    for seed in range(20):
        train = noisy_decay(n, seed)
        grad = np.random.default_rng(1000 + seed).lognormal(0.0, 0.5, n)
        run = make_run(**{"train/loss": train.tolist(), "grad_norm": grad.tolist()})
        alarms += any(f.severity != "info" for f in run_all(run))
    assert alarms == 0


def test_points_far_out_in_mad_terms_but_small_next_to_the_run_are_not_spikes() -> None:
    # The public-log pattern: a loss that fell from 0.07 to 0.005, with one point at 0.014.
    n = 3000
    level = 0.005 + 0.065 * np.exp(-np.arange(n) / 400)
    values = level * np.random.default_rng(3).lognormal(0.0, 0.08, n)
    values[2400] = 0.0141
    findings = SpikeDetector().run(make_run(**{"train/loss": values.tolist()}))
    # Unusual for so smooth a run, and 14% of its progress: worth a note, not an alarm.
    assert [(f.title, f.severity) for f in findings] == [("Isolated loss spikes", "info")]
    assert findings[0].evidence["count"] == 1


def test_points_a_noisy_run_produces_by_chance_are_not_spikes() -> None:
    # Per-step scatter of 25%: a point at twice the level is three sigma, and a few
    # thousand points will contain several. None of them is an event.
    for seed in range(20):
        values = noisy_decay(8000, seed)
        assert SpikeDetector().run(make_run(**{"train/loss": values.tolist()})) == [], seed


def test_downward_points_are_never_spikes() -> None:
    values = noisy_decay(500, seed=4, sigma=0.05)
    values[300] = 1e-6
    assert SpikeDetector().run(make_run(**{"train/loss": values.tolist()})) == []


def test_flat_noise_has_no_progress_to_measure_against() -> None:
    # A reward-driven objective logged as "loss": it never falls, so nothing about
    # levels can be said. Checked over many seeds because "best" is a minimum.
    clear = 0
    for seed in range(200):
        values = 0.5 + np.random.default_rng(seed).normal(0.0, 0.2, 400)
        lv = levels(values)
        assert lv is not None
        clear += lv.has_progress
    assert clear <= 4  # 2% of 200


# ---------------------------------------------------------------------------
# What costs a run is reported, with what it cost.
# ---------------------------------------------------------------------------


def test_a_spike_that_gives_back_most_of_the_progress_is_a_warning() -> None:
    values = noisy_decay(1000, seed=5, sigma=0.05)
    values[700] = 1.9  # back near the starting level for one step
    findings = SpikeDetector().run(make_run(**{"train/loss": values.tolist()}))
    assert [f.title for f in findings] == ["Loss spike"]
    assert findings[0].severity == "warning"
    assert findings[0].step_start == 700
    assert findings[0].evidence["share_of_progress"] > 0.8
    assert "of the run's progress" in findings[0].message


def test_many_spikes_become_one_finding() -> None:
    values = noisy_decay(2000, seed=6, sigma=0.05)
    for step in range(600, 2000, 150):
        values[step] = 2.5
    findings = SpikeDetector().run(make_run(**{"train/loss": values.tolist()}))
    assert [f.title for f in findings] == ["Repeated loss spikes"]
    assert findings[0].evidence["count"] == 10


def test_a_level_that_climbs_and_returns_is_a_setback_with_its_cost() -> None:
    # Up threefold for 300 of 3,000 steps, then back: a tenth of the run.
    values = with_bump(noisy_decay(3000, seed=7), start=1500, length=300, factor=3.0)
    run = make_run(**{"train/loss": values.tolist()})
    findings = SetbackDetector().run(run)
    assert [f.title for f in findings] == ["Setback"]
    setback = findings[0]
    assert setback.severity == "warning"
    assert 1400 <= setback.step_start <= 1560 and 1750 <= setback.step_end <= 1900
    assert 0.07 <= setback.evidence["share_of_run"] <= 0.15
    # It came back, so it is not a divergence, and no single point stood out.
    assert DivergenceDetector().run(run) == []
    assert SpikeDetector().run(run) == []


def test_a_short_setback_is_info_and_a_blip_is_nothing() -> None:
    # 500 of 20,000 steps: 2.5% of the run, and longer than the smoothing window.
    base = noisy_decay(20000, seed=8)
    short = SetbackDetector().run(
        make_run(**{"train/loss": with_bump(base, 12000, 500, 3.0).tolist()})
    )
    assert [f.severity for f in short] == ["info"]
    assert 0.015 <= short[0].evidence["share_of_run"] <= 0.035
    blip = SetbackDetector().run(
        make_run(**{"train/loss": with_bump(base, 12000, 40, 3.0).tolist()})
    )
    assert blip == []


def test_three_high_points_in_a_short_log_are_a_burst() -> None:
    # A log of 100 points is smoothed over five. Three high points in a row are most of
    # that window: the median rises with them, so no point stands out from it, and the
    # stretch is shorter than the window, so it was not a setback either. On the public
    # logs of `studies/hub-census/` such a burst went unreported at any height.
    base = noisy_decay(100, seed=11, sigma=0.05)
    burst = base.copy()
    burst[50:53] += 2.0 - 0.2  # the run's whole progress
    run = make_run(**{"train/loss": burst.tolist()})
    assert SpikeDetector().run(run) == []
    findings = SetbackDetector().run(run)
    assert [(f.title, f.severity) for f in findings] == [("Loss burst", "warning")]
    assert findings[0].step_start == 50 and findings[0].step_end == 53
    assert findings[0].evidence["points"] == 3
    assert 0.8 <= findings[0].evidence["share_of_progress"] <= 1.2
    assert "3 logged points" in findings[0].message
    assert SetbackDetector().run(make_run(**{"train/loss": base.tolist()})) == []


def test_a_burst_must_be_tall_and_stand_out_from_the_scatter() -> None:
    quiet = noisy_decay(100, seed=11, sigma=0.05)
    low = quiet.copy()
    low[50:53] += 0.25 * (2.0 - 0.2)  # a quarter of the progress: under the half it takes
    assert SetbackDetector().run(make_run(**{"train/loss": low.tolist()})) == []
    # The same height in a run whose points scatter by half their level is not unusual.
    noisy = noisy_decay(100, seed=11, sigma=0.5)
    lifted = noisy.copy()
    lifted[50:53] += 0.6 * (2.0 - 0.2)
    assert SetbackDetector().run(make_run(**{"train/loss": lifted.tolist()})) == []


def test_a_burst_of_three_percent_of_a_long_run_is_reported_once() -> None:
    # 1,000 points are smoothed over 41. Thirty high points lift the median and are fewer
    # than the window.
    values = with_bump(noisy_decay(1000, seed=12, sigma=0.1), start=600, length=30, factor=6.0)
    run = make_run(**{"train/loss": values.tolist()})
    findings = SetbackDetector().run(run)
    assert [f.title for f in findings] == ["Loss burst"]
    assert 590 <= findings[0].step_start <= 610 and 620 <= findings[0].step_end <= 640
    assert SpikeDetector().run(run) == []


def test_many_bursts_are_one_finding_that_counts_them() -> None:
    # A public run of 20,000 points blew up and came back 47 times. Forty-seven findings
    # would bury everything else in the report.
    # 400 points are smoothed over 17, so twelve high points in a row lift the level.
    values = noisy_decay(400, seed=13, sigma=0.05)
    for start in range(120, 400, 40):
        values[start : start + 12] += 2.0 - 0.2
    findings = SetbackDetector().run(make_run(**{"train/loss": values.tolist()}))
    assert [(f.title, f.severity) for f in findings] == [("Repeated loss bursts", "warning")]
    assert findings[0].evidence["count"] == 7
    assert findings[0].step_start <= 121 and findings[0].step_end >= 360
    assert "7 times" in findings[0].message


def test_a_run_that_ends_inside_a_rise_is_judged_by_what_it_gave_back() -> None:
    base = noisy_decay(2000, seed=9, sigma=0.05)
    progress = 2.0 - base[-200:].mean()
    lost_most = base.copy()
    lost_most[1700:] += 0.8 * progress
    lost_some = base.copy()
    lost_some[1700:] += 0.35 * progress

    error = DivergenceDetector().run(make_run(**{"train/loss": lost_most.tolist()}))
    assert [(f.title, f.severity) for f in error] == [("Diverged (did not recover)", "error")]
    assert 0.6 <= error[0].evidence["gave_back"] <= 1.0
    assert 1650 <= error[0].step_start <= 1760

    warning = DivergenceDetector().run(make_run(**{"train/loss": lost_some.tolist()}))
    assert [(f.title, f.severity) for f in warning] == [("Diverged (did not recover)", "warning")]

    # The setback detector leaves an unrecovered rise to the divergence detector.
    assert SetbackDetector().run(make_run(**{"train/loss": lost_most.tolist()})) == []


def test_a_loss_that_only_ever_rises_is_an_error() -> None:
    values = [2.5 * 1.08**step for step in range(1, 61)]
    findings = DivergenceDetector().run(make_run(**{"train/loss": values}))
    assert [(f.title, f.severity) for f in findings] == [
        ("Diverged (rose from the start)", "error")
    ]


def test_a_rising_evaluation_loss_is_not_called_divergence() -> None:
    # Training loss to zero, evaluation loss doubling: overfitting, with a different remedy.
    n = 400
    train = noisy_decay(n, seed=10, floor=0.01, sigma=0.05)
    evaluation = [1.0 - 0.5 * min(i, 5) / 5 + 0.03 * max(0, i - 5) for i in range(40)]
    run = make_run(**{"train/loss": train.tolist()})
    for i, value in enumerate(evaluation):
        run.add_point("eval/loss", i * 10, value)
    run.finalize()
    by_detector = {f.detector: f for f in run_all(run)}
    assert "divergence" not in by_detector
    assert by_detector["overfitting"].title == "Overfitting onset"
    assert by_detector["overfitting"].severity == "error"


# ---------------------------------------------------------------------------
# Overfitting: severity follows the size of the rise.
# ---------------------------------------------------------------------------


def _eval_run(rise_per_eval: float, evals_after_best: int, with_train: bool = True) -> list:
    evaluation = [1.0, 0.8, 0.7, 0.65] + [
        0.65 + rise_per_eval * (i + 1) for i in range(evals_after_best)
    ]
    metrics = {"eval/loss": evaluation}
    if with_train:
        metrics["train/loss"] = [1.0 * 0.9**i + 0.05 for i in range(len(evaluation))]
    return OverfittingDetector().run(make_run(**metrics))


def test_overfitting_severity_follows_the_rise() -> None:
    assert _eval_run(0.002, 4) == []  # 1.2% above its best: within what a small eval set does
    assert [f.severity for f in _eval_run(0.008, 4)] == ["info"]  # 4.9%
    assert [f.severity for f in _eval_run(0.03, 4)] == ["warning"]  # 18%
    assert [f.severity for f in _eval_run(0.1, 4)] == ["error"]  # 62%


def test_a_small_rise_with_an_unmistakable_trend_is_still_a_warning() -> None:
    findings = _eval_run(0.004, 12)  # 7.4% over twelve evaluations, every one higher
    assert [f.severity for f in findings] == ["warning"]


def test_a_run_with_three_evaluations_can_still_be_told_it_ended_far_above_its_best() -> None:
    # The slope test needs three evaluations after the best one. Many real runs evaluate
    # three or four times in all.
    doubled = OverfittingDetector().run(make_run(**{"eval/loss": [1.0, 0.5, 1.0]}))
    assert [(f.severity, f.title) for f in doubled] == [("error", "Evaluation loss rising")]
    assert "too few for a trend test" in doubled[0].message

    quarter = OverfittingDetector().run(make_run(**{"eval/loss": [1.0, 0.5, 0.58, 0.65]}))
    assert [f.severity for f in quarter] == ["warning"]

    one_high_reading = OverfittingDetector().run(make_run(**{"eval/loss": [1.0, 0.5, 0.51, 0.65]}))
    assert one_high_reading == []


def test_without_a_training_loss_no_cause_is_claimed() -> None:
    findings = _eval_run(0.03, 4, with_train=False)
    assert [f.title for f in findings] == ["Evaluation loss rising"]
    assert "train/loss" not in findings[0].message


# ---------------------------------------------------------------------------
# Gradient norm and learning rate: observations, not alarms.
# ---------------------------------------------------------------------------


def test_heavy_tailed_gradient_norms_are_not_explosions() -> None:
    norms = np.random.default_rng(11).lognormal(0.0, 0.5, 5000)
    assert GradNormDetector().run(make_run(grad_norm=norms.tolist())) == []


def test_real_explosions_are_counted_in_one_info_finding() -> None:
    norms = np.random.default_rng(12).lognormal(0.0, 0.3, 2000)
    norms[[400, 900, 1500]] = [80.0, 300.0, 55.0]
    findings = GradNormDetector().run(make_run(grad_norm=norms.tolist()))
    assert [(f.title, f.severity) for f in findings] == [("Gradient-norm explosion", "info")]
    assert findings[0].evidence["count"] == 3
    assert findings[0].evidence["peak_step"] == 900


def test_a_non_finite_gradient_norm_is_a_note_while_the_loss_stays_finite() -> None:
    norms = [1.0] * 50
    norms[20] = math.nan
    findings = GradNormDetector().run(make_run(grad_norm=norms))
    assert [(f.title, f.severity) for f in findings] == [("Non-finite gradient norm", "info")]


def test_a_nan_in_the_training_loss_is_an_error_and_in_an_evaluation_a_warning() -> None:
    train = noisy_decay(200, seed=15, sigma=0.05).tolist()
    evaluation = [1.0, 0.8, math.nan, 0.6, 0.55]
    broken_eval = DivergenceDetector().run(
        make_run(**{"train/loss": train, "eval/loss": evaluation})
    )
    assert [(f.title, f.severity) for f in broken_eval] == [
        ("Non-finite evaluation values", "warning")
    ]

    train[120] = math.inf
    broken_train = DivergenceDetector().run(make_run(**{"train/loss": train}))
    assert [(f.title, f.severity) for f in broken_train] == [
        ("Diverged (non-finite values)", "error")
    ]


def test_a_coarsely_logged_warmup_is_not_a_discontinuity() -> None:
    # Ten warmup steps logged every five: two uneven jumps to the peak, then a linear decay.
    lr = [5e-6, 1e-5] + [1e-5 * (1 - i / 60) for i in range(1, 60)]
    findings = LRScheduleDetector().run(make_run(lr=lr))
    assert [f.title for f in findings if "discontinuity" in f.title.lower()] == []


def test_a_jump_after_the_lr_has_decayed_is_still_a_discontinuity() -> None:
    lr = [1e-4 * i / 10 for i in range(1, 11)] + [1e-4 * 0.97**i for i in range(60)]
    lr[50] = 5e-4
    titles = [f.title for f in LRScheduleDetector().run(make_run(lr=lr))]
    assert "LR discontinuity" in titles


def test_no_warmup_is_an_observation() -> None:
    lr = [1e-4 * (1 - i / 100) for i in range(100)]
    findings = LRScheduleDetector().run(make_run(lr=lr))
    assert [(f.title, f.severity) for f in findings] == [("No LR warmup detected", "info")]


# ---------------------------------------------------------------------------
# The level helpers themselves.
# ---------------------------------------------------------------------------


def test_levels_read_a_noisy_curve() -> None:
    lv = levels(noisy_decay(4000, seed=13))
    assert lv is not None and lv.has_progress
    assert 1.4 <= lv.start <= 2.8  # three noisy points near the true 2.0
    assert 0.17 <= lv.best <= 0.24
    assert abs(lv.final - lv.best) < 0.05
    assert lv.window == 161  # n // 25, made odd


def test_excursions_are_found_once_and_marked_recovered_or_not() -> None:
    base = noisy_decay(3000, seed=14, sigma=0.05)
    lv = levels(with_bump(base, 1000, 400, 2.5))
    assert lv is not None
    found = excursions(lv)
    assert [e.recovered for e in found] == [True]
    assert 950 <= found[0].start_index <= 1060 and 1350 <= found[0].end_index <= 1500

    tail = base.copy()
    tail[2600:] *= 4.0  # from 0.2 to 0.8: a third of the progress from 2.0
    lv_tail = levels(tail)
    assert lv_tail is not None
    assert [e.recovered for e in excursions(lv_tail)] == [False]


def test_too_few_points_have_no_levels() -> None:
    assert levels(np.array([1.0, 0.9, 0.8])) is None
