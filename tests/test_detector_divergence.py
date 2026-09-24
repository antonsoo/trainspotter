from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.divergence import DivergenceDetector


def test_nan_is_flagged_as_error() -> None:
    values = [1.0, 0.9, 0.8, float("nan"), 0.7, 0.6]
    run = make_run(**{"train/loss": values})

    findings = DivergenceDetector(metrics=("train/loss",)).run(run)

    nan_findings = [f for f in findings if "non-finite" in f.title.lower()]
    assert len(nan_findings) == 1
    assert nan_findings[0].severity == "error"
    assert nan_findings[0].step_start == 3


def test_sustained_growth_is_flagged() -> None:
    # decays, bottoms out, then climbs steadily for the whole tail window
    decay = [5.0 * 0.9**i + 0.5 for i in range(30)]
    growth = [decay[-1] + 0.3 * i for i in range(1, 30)]
    values = decay + growth
    run = make_run(**{"train/loss": values})

    findings = DivergenceDetector(metrics=("train/loss",)).run(run)

    growth_findings = [f for f in findings if "growth" in f.title.lower()]
    assert len(growth_findings) == 1
    assert growth_findings[0].severity == "error"


def test_clean_decay_has_no_false_alarm() -> None:
    values = [5.0 * 0.95**i + 0.1 for i in range(100)]
    run = make_run(**{"train/loss": values})

    findings = DivergenceDetector(metrics=("train/loss",)).run(run)

    assert findings == []


def test_rising_accuracy_metric_not_checked_by_default() -> None:
    # DivergenceDetector's default metrics are train/loss and eval/loss only;
    # a metric that's supposed to increase (like accuracy) simply isn't looked at.
    values = [0.1 * i for i in range(20)]
    run = make_run(**{"eval/accuracy": values})

    findings = DivergenceDetector().run(run)

    assert findings == []
