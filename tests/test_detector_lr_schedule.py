from __future__ import annotations

from conftest import make_run

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
