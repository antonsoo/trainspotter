# Captured timing and changing logging cadence

These are controlled experiments, not production training logs or hardware
benchmarks. [`capture_timing.py`](../capture_timing.py) performs two 400-update
logistic-regression runs on the same synthetic 128-by-8 dataset (seed 20261009).
Losses are computed, and timestamps and update durations are measured with
`time.perf_counter()` in seconds. They are monotonic clock readings, not Unix dates.

Both runs request a minimum update duration of 3 ms using sleep. One increases
that request to 9 ms after update 200. The logger records every fourth update
through 200, then every tenth, plus an initial observation at step 0.
[`capture.json`](capture.json) retains every update's observations, environment,
injections, CSV counts and hashes.

| File | Contents |
| --- | --- |
| `stable-with-steps.csv` | 71 observations with steps, timestamps and loss; changing logging cadence |
| `stable-without-steps.csv` | Identical observed loss/timestamps with the step column removed |
| `slow-with-steps.csv` | 71 observations from the run with an injected late delay |
| `slow-direct.csv` | All 400 measured update durations and losses, without a step column |
| `slow-throughput.csv` | The same durations expressed as 128 samples divided by seconds; no step column |
| `slow-missing-time.csv` | The delayed run with the timestamp at step 16 removed |
| `slow-broken-clock.csv` | The delayed run with the step-16 timestamp deliberately moved before step 12 |

Capture fresh inputs without replacing these files, then verify them:

```sh
uv run python examples/capture_timing.py --output /tmp/trainspotter-timing
uv run python scripts/verify_timing.py --capture /tmp/trainspotter-timing \
  --output /tmp/trainspotter-timing/verification.json
```

Actual durations vary with the machine and scheduler. The verifier calculates
expected medians from the recorded timestamps and counts, then checks real CLI
reports and both ordinary and timing-required gate outcomes. It fails if a new
capture does not exhibit the intended control and delay outcomes.

The [timing guide](../../docs/timing-evidence.md) explains the findings and limits.
