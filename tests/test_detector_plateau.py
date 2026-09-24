from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.plateau import PlateauDetector


def test_a_genuine_stall_is_detected() -> None:
    # Decays to ~5, stalls there for 60 steps, then decays much further to
    # ~0.5 -- the stall is NOT near the run's eventual floor (there's a lot
    # more improvement still to come), so it must be flagged, unlike a
    # plateau that sits at the run's actual floor (see
    # test_convergence_near_the_floor_is_not_flagged below).
    decay_to_stall = [5.0 * 0.9**i + 5.0 for i in range(40)]  # 10 -> ~5
    stall = [5.0 + 0.001 * ((-1) ** i) for i in range(60)]
    decay_further = [5.0 * 0.9**i + 0.5 for i in range(40)]  # ~5.5 -> ~0.5
    values = decay_to_stall + stall + decay_further
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert len(findings) == 1
    assert findings[0].step_start == 40
    assert findings[0].step_end == 99  # the stall, not the later decay


def test_a_brief_blip_does_not_fragment_one_plateau_into_many() -> None:
    # A near-constant region with a short-lived bump in the middle: windows
    # overlapping the bump fail the rel_change guard, so the flagged
    # window-starts come in two separate clusters with a gap between them.
    # Found on the real overfitting example, where near-zero train/loss
    # noise did exactly this and produced 17 near-duplicate findings before
    # the merge-gap fix. Uses the same "more decay comes later" shape as
    # test_a_genuine_stall_is_detected so convergence suppression doesn't
    # also swallow it.
    decay_to_stall = [5.0 * 0.9**i + 5.0 for i in range(40)]
    stall = [5.0 + 0.001 * ((-1) ** i) for i in range(60)]
    for i in range(28, 33):
        stall[i] += 0.5
    decay_further = [5.0 * 0.9**i + 0.5 for i in range(40)]
    values = decay_to_stall + stall + decay_further
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert len(findings) == 1
    assert findings[0].step_start == 40
    assert findings[0].step_end == 99


def test_convergence_near_the_floor_is_not_flagged() -> None:
    # Decays and then flattens out AT its eventual floor (nothing lower
    # comes later) -- ordinary end-of-training convergence, the common
    # case on a real cosine/warmup schedule. Must not be reported.
    decay = [5.0 * 0.9**i + 1.0 for i in range(40)]
    flat = [1.001, 0.999, 1.0, 1.002, 0.998] * 12  # ~60 points barely moving
    values = decay + flat
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert findings == []


def test_flat_window_with_decayed_lr_is_not_flagged_even_if_not_near_floor() -> None:
    # Same "more decay comes later" shape as test_a_genuine_stall_is_detected
    # (so the achieved-fraction check alone would NOT suppress this), but
    # with an `lr` series that's already decayed to well under 30% of its
    # peak by the stall -- the second, independent convergence signal.
    decay_to_stall = [5.0 * 0.9**i + 5.0 for i in range(40)]
    stall = [5.0 + 0.001 * ((-1) ** i) for i in range(60)]
    decay_further = [5.0 * 0.9**i + 0.5 for i in range(40)]
    values = decay_to_stall + stall + decay_further
    lr = [1.0] * 10 + [0.05] * (len(values) - 10)  # decayed well before the stall starts
    run = make_run(**{"train/loss": values, "lr": lr})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert findings == []


def test_steadily_improving_curve_has_no_plateau() -> None:
    # Linear decay at a constant, substantial rate throughout -- unlike an
    # exponential decay, it never flattens out near an asymptote, so no
    # window should look "flat" by either the slope test or rel_change.
    values = [10.0 - 0.05 * i for i in range(150)]
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert findings == []
