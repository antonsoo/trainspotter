from __future__ import annotations

import random

from conftest import make_run

from trainspotter.detectors.eval_noise import EvalNoiseDetector


def test_noisy_eval_relative_to_small_total_improvement_is_flagged() -> None:
    rng = random.Random(1)
    # true improvement is tiny (0.9 -> 0.85) but jitter per step is comparable in size
    values = [0.9 - 0.05 * (i / 19) + rng.gauss(0, 0.03) for i in range(20)]
    run = make_run(**{"eval/loss": values})

    findings = EvalNoiseDetector().run(run)

    assert len(findings) == 1


def test_smooth_large_improvement_is_not_flagged() -> None:
    values = [2.0 * 0.9**i + 0.1 for i in range(30)]
    run = make_run(**{"eval/loss": values})

    findings = EvalNoiseDetector().run(run)

    assert findings == []


def test_six_smooth_evaluations_do_not_measure_noise() -> None:
    # The previous detector called even exact linear improvement "20% noise".
    for values in ([1, 0.9, 0.8, 0.7, 0.6, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        assert EvalNoiseDetector().run(make_run(**{"eval/loss": values})) == []


def test_one_turn_is_a_valley_or_peak_not_repeated_fluctuation() -> None:
    for values in ([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]):
        assert EvalNoiseDetector().run(make_run(**{"eval/loss": values})) == []


def test_fluctuation_is_descriptive_and_scale_invariant() -> None:
    import math

    for scale in (1e-200, 1.0, 1e308):
        values = [scale * x for x in (1, -1, 1, -1, 1, -1)]
        (finding,) = EvalNoiseDetector().run(make_run(**{"eval/loss": values}))
        assert finding.title == "Evaluation values fluctuate"
        assert finding.severity == "info"
        assert finding.evidence["direction_changes"] == 4
        assert finding.evidence["ratio"] == 1
        assert all(math.isfinite(v) for v in finding.evidence.values() if isinstance(v, float))
        assert "does not measure evaluation uncertainty" in finding.message


def test_nonfinite_readings_break_variation_evidence() -> None:
    import math

    for invalid in (math.nan, math.inf, -math.inf):
        short = make_run(**{"eval/loss": [1.0, 2.0, 1.0, invalid, 2.0, 1.0, 2.0]})
        assert EvalNoiseDetector().run(short) == []
        long = make_run(**{"eval/loss": [1.0, 2.0] * 3 + [invalid] + [2.0, 1.0] * 3})
        findings = EvalNoiseDetector().run(long)
        assert [(f.step_start, f.step_end) for f in findings] == [(0, 5), (7, 12)]
        assert [f.evidence["direction_changes"] for f in findings] == [4, 4]


def test_plateaus_do_not_create_directions() -> None:
    # Ignoring zero changes preserves actual turns across plateaus; zero changes
    # still enter the median so long flat runs do not become high-variation notes.
    (finding,) = EvalNoiseDetector().run(make_run(**{"eval/loss": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]}))
    assert finding.evidence["direction_changes"] == 2
    assert EvalNoiseDetector().run(make_run(**{"eval/loss": [0.0] * 6})) == []
