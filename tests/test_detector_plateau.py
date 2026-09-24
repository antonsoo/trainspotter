from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.plateau import PlateauDetector


def test_flat_segment_is_detected() -> None:
    decay = [5.0 * 0.9**i + 1.0 for i in range(40)]
    flat = [1.001, 0.999, 1.0, 1.002, 0.998] * 12  # ~60 points barely moving
    values = decay + flat
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert len(findings) >= 1
    # the flagged region should overlap the flat tail, not the decaying head
    assert findings[0].step_end >= len(decay)


def test_steadily_improving_curve_has_no_plateau() -> None:
    # Linear decay at a constant, substantial rate throughout -- unlike an
    # exponential decay, it never flattens out near an asymptote, so no
    # window should look "flat" by either the slope test or rel_change.
    values = [10.0 - 0.05 * i for i in range(150)]
    run = make_run(**{"train/loss": values})

    findings = PlateauDetector(metrics=("train/loss",), min_window=20).run(run)

    assert findings == []
