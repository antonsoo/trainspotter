from __future__ import annotations

from trainspotter.detectors.throughput import ThroughputDetector
from trainspotter.model import Run


def _run_with_step_time(values: list[float]) -> Run:
    run = Run(source_format="synthetic", source_path="<test>")
    for step, value in enumerate(values):
        run.add_point("step_time", step, value)
    run.finalize()
    return run


def test_regression_detected_when_recent_steps_are_much_slower() -> None:
    values = [0.10] * 20 + [0.25] * 20  # 2.5x slower in the recent window
    run = _run_with_step_time(values)

    findings = ThroughputDetector().run(run)

    assert len(findings) == 1
    assert findings[0].evidence["ratio"] >= 1.4


def test_stable_throughput_has_no_false_alarm() -> None:
    values = [0.10 + 0.005 * ((-1) ** i) for i in range(40)]
    run = _run_with_step_time(values)

    findings = ThroughputDetector().run(run)

    assert findings == []
