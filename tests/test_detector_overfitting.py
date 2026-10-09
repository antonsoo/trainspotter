from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.overfitting import OverfittingDetector


def test_onset_detected_when_eval_rises_after_train_keeps_falling() -> None:
    n = 60
    best_step = 30
    train = [2.0 * 0.95**i + 0.05 for i in range(n)]
    eval_loss = [2.2 * 0.95**i + 0.1 for i in range(best_step)]
    eval_loss += [eval_loss[-1] + 0.03 * (i - best_step + 1) for i in range(best_step, n)]
    run = make_run(**{"train/loss": train, "eval/loss": eval_loss})

    findings = OverfittingDetector().run(run)

    assert len(findings) == 1
    assert findings[0].evidence["best_step"] == best_step - 1  # index of the minimum
    # It ends at more than twice its best: the last checkpoint is far worse than the best one.
    assert findings[0].evidence["relative_rise"] > 0.5
    assert findings[0].severity == "error"
    assert findings[0].title == "Possible overfitting"


def test_both_still_improving_has_no_finding() -> None:
    n = 60
    train = [2.0 * 0.95**i + 0.05 for i in range(n)]
    eval_loss = [2.2 * 0.96**i + 0.08 for i in range(n)]
    run = make_run(**{"train/loss": train, "eval/loss": eval_loss})

    findings = OverfittingDetector().run(run)

    assert findings == []


def test_later_training_cannot_supply_missing_overlap() -> None:
    run = make_run(**{"eval/loss": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    (before,) = OverfittingDetector().run(run)
    for step in range(100, 151):
        run.add_point("train/loss", step, 2.0 - step / 100)
    run.finalize()
    (after,) = OverfittingDetector().run(run)
    assert before.title == after.title == "Evaluation loss rising"
    assert after.evidence["training_status"] == "insufficient_overlap"
    assert after.evidence["training_observations"] == 0
    assert after.message == before.message


def test_future_training_cannot_reverse_contemporaneous_evidence() -> None:
    run = make_run(
        **{
            "eval/loss": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "train/loss": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    before = OverfittingDetector().run(run)
    for step in range(6, 200):
        run.add_point("train/loss", step, 0.6 * (200 - step) / 194)
    run.finalize()
    assert OverfittingDetector().run(run) == before
    assert before[0].title == "Evaluation loss rising"


def test_eval_must_rise_within_the_training_overlap() -> None:
    # Evaluation gets worse AFTER a brief interval when both losses declined.
    run = make_run(**{"eval/loss": [0.5, 1.0, 0.9, 0.8, 1.0, 1.5, 2.0]})
    for step, loss in [(1, 1.0), (2, 0.8), (3, 0.6)]:
        run.add_point("train/loss", step, loss)
    run.finalize()
    (finding,) = OverfittingDetector().run(run)
    assert finding.title == "Evaluation loss rising"
    assert finding.evidence["training_status"] == "opposing_trends_not_established"
    assert finding.evidence["overlap_eval_scaled_slope"] < 0


def test_overlapping_trends_retain_source_bounds() -> None:
    run = make_run(
        **{
            "eval/loss": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "train/loss": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    (finding,) = OverfittingDetector().run(run)
    assert finding.title == "Possible overfitting"
    assert finding.evidence["training_step_start"] == 0
    assert finding.evidence["training_step_end"] == 5
    assert finding.evidence["training_observations"] == 6
    assert finding.evidence["overlap_eval_observations"] == 6
    assert "does not establish its cause" in finding.message
    assert finding.fixes[0].startswith("If a checkpoint was saved")


def test_invalid_training_cannot_supply_a_slope() -> None:
    run = make_run(
        **{
            "eval/loss": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "train/loss": [1.0, 0.9, float("nan"), 0.7, 0.6, 0.5],
        }
    )
    (finding,) = OverfittingDetector().run(run)
    assert finding.evidence["training_status"] == "nonfinite_overlap"
    assert finding.title == "Evaluation loss rising"


def test_only_the_last_finite_eval_segment_is_current() -> None:
    for invalid in (float("nan"), float("inf"), -float("inf")):
        run = make_run(**{"eval/loss": [0.1, 0.2, invalid, 1.0, 1.1, 1.2, 1.3, 1.4]})
        (finding,) = OverfittingDetector().run(run)
        assert finding.evidence["best_value"] == 1.0
        assert finding.step_start == finding.evidence["segment_start"] == 3
        assert finding.step_end == 7
        assert "after the last non-finite evaluation" in finding.message
        assert OverfittingDetector().run(make_run(**{"eval/loss": [1.0, 2.0, 3.0, invalid]})) == []


def test_large_final_rise_does_not_claim_a_missing_trend_test() -> None:
    (finding,) = OverfittingDetector().run(
        make_run(**{"eval/loss": [1.0, 10.0, 8.0, 6.0, 4.0, 2.0]})
    )
    assert finding.evidence["slope_status"] == "assessed"
    assert "slope criterion not met" in finding.message
    assert "too few" not in finding.message


def test_negative_baseline_requires_a_real_penultimate_rise() -> None:
    assert OverfittingDetector().run(make_run(**{"eval/loss": [-1.0, -0.99, -0.74]})) == []
    (finding,) = OverfittingDetector().run(make_run(**{"eval/loss": [-1.0, -0.85, -0.74]}))
    assert finding.severity == "warning"
    assert "26.0% of the absolute minimum" in finding.message


def test_zero_baseline_does_not_invent_a_percentage_or_error() -> None:
    (finding,) = OverfittingDetector().run(make_run(**{"eval/loss": [0.0, 0.01, 0.02, 0.03]}))
    assert finding.severity == "info"
    assert finding.evidence["relative_rise"] == "unavailable: zero baseline"
    assert "relative rise unavailable" in finding.message


def test_rise_thresholds_do_not_depend_on_units() -> None:
    for scale in (1e-200, 1.0, 1e300):
        (finding,) = OverfittingDetector().run(
            make_run(**{"eval/loss": [scale * x for x in (1.0, 1.2, 1.4, 1.6)]})
        )
        assert finding.severity == "error"
        assert finding.evidence["slope_p_value"] < 0.001
