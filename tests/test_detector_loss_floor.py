from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.loss_floor import LossFloorDetector


def test_implausibly_fast_drop_is_flagged() -> None:
    values = [2.3] + [0.001] * 4 + [2.3 * 0.97**i for i in range(5, 100)]
    run = make_run(**{"train/loss": values})

    findings = LossFloorDetector().run(run)

    assert len(findings) == 1
    assert findings[0].severity == "info"


def test_normal_gradual_descent_is_not_flagged() -> None:
    values = [2.3 * 0.98**i + 0.1 for i in range(100)]
    run = make_run(**{"train/loss": values})

    findings = LossFloorDetector().run(run)

    assert findings == []
