from __future__ import annotations

from conftest import make_run

from trainspotter.detectors.spikes import SpikeDetector


def test_planted_spike_is_found_near_its_step() -> None:
    n = 100
    values = [2.0 * 0.97**i + 0.02 for i in range(n)]
    spike_step = 60
    values[spike_step] += 5.0  # a clear, large spike
    run = make_run(**{"train/loss": values})

    findings = SpikeDetector(metrics=("train/loss",)).run(run)

    assert len(findings) == 1
    f = findings[0]
    assert f.step_start <= spike_step <= f.step_end
    assert f.severity in ("warning", "error")
    assert f.evidence["peak_step"] == spike_step


def test_smooth_decay_has_no_false_alarm() -> None:
    values = [2.0 * 0.97**i + 0.02 for i in range(100)]
    run = make_run(**{"train/loss": values})

    findings = SpikeDetector(metrics=("train/loss",)).run(run)

    assert findings == []


def test_mild_noise_does_not_trigger_default_threshold() -> None:
    import random

    rng = random.Random(0)
    values = [2.0 * 0.97**i + 0.02 + rng.gauss(0, 0.01) for i in range(200)]
    run = make_run(**{"train/loss": values})

    findings = SpikeDetector(metrics=("train/loss",)).run(run)

    assert findings == []
