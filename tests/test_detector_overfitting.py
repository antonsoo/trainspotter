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
    assert findings[0].severity == "warning"


def test_both_still_improving_has_no_finding() -> None:
    n = 60
    train = [2.0 * 0.95**i + 0.05 for i in range(n)]
    eval_loss = [2.2 * 0.96**i + 0.08 for i in range(n)]
    run = make_run(**{"train/loss": train, "eval/loss": eval_loss})

    findings = OverfittingDetector().run(run)

    assert findings == []
