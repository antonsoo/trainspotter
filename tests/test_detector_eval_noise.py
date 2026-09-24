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
