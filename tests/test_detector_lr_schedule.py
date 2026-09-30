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


def test_short_warmup_logged_every_few_steps_is_not_a_discontinuity() -> None:
    # The real Hugging Face run in examples/: 20 warmup steps logged every 5 steps, so each logged
    # interval covers a quarter of the LR range - a steady ramp, not a jump.
    from conftest import make_run_steps

    steps = list(range(5, 730, 5))
    peak = 5e-4
    values = [min(peak, peak * s / 20) if s <= 20 else peak * (1 - (s - 20) / 710) for s in steps]
    findings = LRScheduleDetector().run(make_run_steps("lr", steps, values))
    assert [f.title for f in findings] == []


def test_the_real_transformers_run_has_no_lr_finding() -> None:
    from pathlib import Path

    from trainspotter.readers import load_run

    run = load_run(
        Path(__file__).parent.parent / "examples" / "transformers_tinygpt2.trainer_state.json"
    )
    assert LRScheduleDetector().run(run) == []


def test_a_jump_over_a_logging_interval_says_so() -> None:
    from conftest import make_run_steps

    steps = list(range(0, 200, 10))
    values = [1e-4] * 10 + [1e-6] * 10  # one step-decay drop between logged points 10 steps apart
    findings = LRScheduleDetector().run(make_run_steps("lr", steps, values))
    jump = next(f for f in findings if f.title == "LR discontinuity")
    assert "across one logged interval (10 steps)" in jump.message
