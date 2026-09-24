from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.grad_norm import GradNormDetector


def test_explosion_is_found_near_its_step() -> None:
    values = [1.0 + 0.05 * ((-1) ** i) for i in range(80)]
    values[50] = 40.0
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    explosions = [f for f in findings if f.title == "Gradient-norm explosion"]
    assert len(explosions) == 1
    assert explosions[0].evidence["peak_step"] == 50


def test_saturation_is_detected_when_pinned_at_a_ceiling() -> None:
    values = [0.5 + 0.05 * i for i in range(20)] + [1.0] * 55
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    assert any(f.title.startswith("Gradient clipping") for f in findings)


def test_stable_norms_have_no_false_alarm() -> None:
    values = [1.0 + 0.02 * ((-1) ** i) for i in range(80)]
    run = make_run(grad_norm=values)

    findings = GradNormDetector().run(run)

    assert findings == []
