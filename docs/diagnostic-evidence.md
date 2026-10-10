# What the loss and gradient histories can establish

The `loss_floor` heuristic now measures "early" using training steps. The
`grad_norm` detector describes concentration without inferring clipping from
a curve alone. Both retain observed non-finite values instead of treating
them as a useful numeric baseline. These changes are in source; they are
not a new PyPI release.

## Logging cadence is not training progress

The capture runs 1,000 float64 CPU PyTorch SGD updates on `loss = w**2 / 2`,
starting at `w = 2`, with learning rate `0.005`. This is a deliberately simple
numerical counterexample, not a model-quality benchmark. Each recorded loss
is independently checked against the closed form `2 * (1 - lr)**(2 * step)`.

The dense CSV retains every observation. The sparse CSV retains step 0 and
steps 900 through 1,000 from **that same run**, without altering any values.

```text
training step    0          100                              900   1000
                 |----------|--------------------------------|------|
early window     [----------]
sparse log       o                                           oooooooo
```

| Input | Before | Current |
| --- | --- | --- |
| Dense history, 1,001 observations | No early-loss finding | No early-loss finding |
| Sparse history, 102 observations | Calls step 900 "early" | No early-loss finding; step 900 is 90% through the span |
| Same sparse readings, step column removed | Calls the second row "early" | Cannot establish training progress from row indices |
| Separate fast run, learning rate `0.1` | Early-loss finding | Retains the finding at step 18, 1.8% through the span |
| Injected infinite starting loss | Calls a finite value 0% of the infinite baseline | No relative-drop claim from an invalid baseline |

The window is the first 10% of the interval between the first and last
logged training steps, inclusive of the boundary. It is not the first 10%
of CSV rows, the full planned training run, or the time since step zero.
An explicit `--step-key` asserts that the selected column is a training
counter. W&B logging counters and inferred row indices cannot supply it.

The first value must be finite and positive. Any negative or non-finite
reading in the early window prevents the heuristic; these observations
cannot support a nonnegative loss floor or an uninterrupted relative drop.
NaN/Inf loss observations remain available to the divergence detector.
The finding retains the exact logged interval and position fraction. It
does not establish data leakage, and a partial/resumed log says nothing
about unobserved training before or after that interval.

## The same pre-clip norms, different clipping behavior

Two further captures optimize `abs(w - 10)` from `w = 0`, with float64 CPU
SGD, learning rate `0.01`, for 100 updates. The gradient magnitude is 1
throughout. One capture never calls a clipping function. The other calls
`torch.nn.utils.clip_grad_norm_` with `max_norm = 0.25` on every update.
The manifest retains measured before/after norms and the function's return
value for every update.

| Recorded quantity | No clipping | Clipping every update |
| --- | --- | --- |
| Pre-clip norm | 1 | 1 |
| Post-clip norm | 1 | About 0.24999975 |
| Returned norm from `clip_grad_norm_` | Function not called | 1 |
| Old finding on pre-clip history | "Gradient clipping likely saturated" | Same claim |
| Current finding on pre-clip history | Observed concentration; clipping unknown | Same description |

PyTorch computes the total norm before applying clipping and returns that
norm; it changes the gradients in place. A return value therefore does not
describe the post-clip norm. See the [PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
and the [2.8.0 implementation used for this capture](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/clip_grad.py).

The concentration note still reports the last 50 observations' fraction
within 0.5% of their observed maximum, from 80%. All 50 must be finite and
nonnegative, with a positive maximum. Existing evidence fields `ceiling`,
`fraction_at_ceiling`, and `window` are retained; `ceiling` means the
**observed maximum**, not a discovered clipping threshold. The added
`clipping_status` is `unknown`. To measure clipping frequency, log the
configured threshold and before/after norms or a clipping indicator.

A single non-finite norm is now reported, even in a one-point log. A
deliberately overflowing float32 `exp(100)` capture exercises this path.
Conflicting repeated-step values prevent trajectory analysis but do not
erase observed NaN/Inf readings. Identical repeated readings count once;
the original input evidence remains in the report.

Non-finite gaps separate the finite segments used for gradient explosions.
A concentration window containing infinity cannot produce the old
"98% within 0.5% of infinity" finding. A separate injected fixture preserves
that exact numerical failure. Neither a gradient explosion nor a flat
gradient curve establishes whether clipping absorbed an update or affected
training quality.

## Inspect and reproduce

- [All recorded and derived inputs](../examples/diagnostic-evidence/), with
  [capture metadata and SHA-256 hashes](../examples/diagnostic-evidence/capture.json).
- [Capture program](../examples/capture_diagnostic_evidence.py): PyTorch
  `2.8.0+cpu`, Python 3.12.12. It uses scalar synthetic objectives and no network.
- [Before/after replay and public-log compatibility audit](../studies/diagnostic-evidence/).
- [Sparse loss report](https://antonsoo.github.io/trainspotter/loss-cadence.html)
  and [unclipped gradient report](https://antonsoo.github.io/trainspotter/gradient-concentration.html).
  Each report embeds its exact input CSV for offline download.

```bash
uv run python studies/diagnostic-evidence/replay.py --check --output /tmp/diagnostics.json
uv run trainspotter analyze examples/diagnostic-evidence/slow-sparse.csv --output html --out /tmp/loss.html
uv run trainspotter analyze examples/diagnostic-evidence/unclipped-before.csv --output json
```

The controlled examples establish counterexamples to the previous claims.
They are not an estimate of field false-positive rates, evidence of a
particular training framework's default logging convention, or a substitute
for inspecting the optimizer and data pipeline in a real run.
