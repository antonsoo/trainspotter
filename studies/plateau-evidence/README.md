# A recovered rise is not one flat loss interval

At source commit `6b4511b`, this loss history received one plateau finding over
all 160 observations, stating that loss "barely moved" with a range of 0.1-4:

```text
steps     0 ... 49 | 50 ... 69 | 70 ... 119 | 120 ... 159
loss         1    |     4     |     1      |     0.1
old       [------------------- plateau ----------------]
current   [plateau]             [plateau]    lower floor
```

The corrected detector retains the two constant stretches, keeps the rise
outside them, and suppresses the final lower floor as convergence under its
existing heuristic. This is a controlled synthetic counterexample. The
[previous finite-gap capture](../finite-segments/README.md) exposed a second
problem: a NaN poisoned the global minimum and disabled convergence suppression.
Its spurious plateau at steps 60-159 is now gone too.

[before.json](before.json) and [after.json](after.json) retain actual CLI findings,
input hashes and gates. [replay.py](replay.py) regenerates nine controlled cases
and replays two derived public loss series. The output is **Possible plateau**;
the message gives the smoothed level and the raw range separately.

| Case | Previous plateau intervals | Corrected intervals |
| --- | --- | --- |
| Fourfold rise, recovery, then lower floor | 0-159 | 0-49 and 70-119 |
| Symmetric excursion with equal endpoints | 14-45 | None |
| Finite segments separated by NaN | 0-29 and 60-159 | 0-29 |
| Exactly 30 constant observations | None | 0-29 |
| Genuine intermediate stall | 40-99 | 40-99 |
| Brief noisy bump during a stall | 51-100 | 51-100 |
| Ordinary converged floor | None | None |
| LR observations only later in the log | Earlier stall hidden | 0-99 |
| Infinite LR between a peak and later value | Later stall hidden | 50-149 |

## What establishes the reported interval

1. Split loss at explicit NaN/Inf readings. The assessment window is 10% of
   that finite segment, at least 30 observations by default. A complete window
   is eligible; 30 observations no longer incorrectly require a 31st.
2. Retain the original endpoint-change and OLS filters. A nonsignificant linear
   slope does not prove a stable level, so these are only candidate filters.
3. Measure a centered-median level. Its smoothing width is half the assessment
   window rounded to odd, at least 3 and at most 201 observations. The allowed
   level range is the larger of 2% of its absolute level or a noise allowance.
4. Estimate the allowance from first differences, so a mixture of distinct
   constant levels cannot inflate it. Use the last `min(window-1, 201)`
   differences in the assessment window:

   ```text
   allowance = 6 * 1.2533 * 1.4826 * MAD(differences) / sqrt(2 * smoothing_width)
   ```

5. Merge touching/overlapping candidate windows only if their entire union
   still meets the level-range bound. Use the smallest noise allowance among
   the merged windows; a noisy interval cannot grant its allowance to a quiet
   one. Preserve the measured range, allowance and qualifying-window count.
6. Apply the existing loss-floor suppression within the finite segment. For
   LR suppression, use only a reading at or before the finding's first step
   and the peak observed up to that reading. NaN/Inf/negative LR resets that
   history; neither a future reading nor a future peak can explain an earlier stall.

The scale factors use normal-reference approximations: the
[MAD normalization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html)
and the median's asymptotic scale in
[Muller, *Possible Advantages of a Robust Evaluation of Comparisons*](https://nvlpubs.nist.gov/nistpubs/jres/105/4/j54mul.pdf).
The factor 6 is a heuristic allowance, consistent with the tool's conservative
robust-scale approach. It is **not a confidence bound** for correlated training
losses, and neither reference validates Trainspotter's detector.

Window min/max use monotone queues with linear work. Both median/MAD widths
are capped at 201; a million-point history does not create a 50,000-point
median window. [Workload measurements](benchmark-after.json) retain time and
process memory, alongside the [previous source](benchmark-before.json).

## Noise controls and public observations

A strict 2% band alone failed a useful independent control: stationary
expected loss with noisy observations. [sensitivity.py](sensitivity.py) generates
a decay, 500 readings with expected loss exactly 5, and a later decay. Across
30 seeds at each of eight relative noise levels from 0% to 20%, the final
detector retains a finding in all 240 cases. It also does so for 240 cases with
separate seeds 100-129. [Before](sensitivity-before.json),
[after](sensitivity-after.json), [separate seeds](sensitivity-heldout.json).

This does not prove performance on real training noise. At 20% IID Gaussian
noise, average coverage of the stationary section is 93.7% in the first set
and 92.0% in the separate seeds. A long noisy stall can split into several
bounded intervals; some controlled cases produce six findings. Median
smoothing can hide a short event, and correlated or changing noise can still
mislead the allowance. These are sensitivity measurements, not accuracy
estimates or a reason to treat a clean report as healthy training.

[public/manifest.json](public/manifest.json) identifies two cached public Trainer
records by repository, revision, source path and hash. Only their unambiguous
numeric step/loss series are reproduced. They are not labelled as healthy or
broken, and their GRPO objectives need not behave like a supervised loss.
One retains its interval with a more precise description of level and raw
scatter. In the other, a previously broad interval narrows and another splits.
Do not interpret these extracts as complete diagnoses: other metrics are omitted.

## Reproduce

From the source checkout:

```sh
uv sync --locked --all-extras --dev
uv run --locked python studies/plateau-evidence/replay.py --artifacts /tmp/plateau-cases --out /tmp/plateau-after.json --verify
uv run --locked trainspotter analyze studies/plateau-evidence/fixtures/returning-rise.csv
```

The replay uses standard-library sorted medians to check each reported level
range independently of the implementation's rolling algorithms. Add
`--executable /path/to/installed/trainspotter` to replay a wheel installation.
The synthetic sensitivity and workload commands are:

```sh
uv run --locked python studies/plateau-evidence/sensitivity.py /tmp/plateau-noise.json --seed-start 100
uv run --locked python studies/plateau-evidence/benchmark.py /tmp/plateau-workloads.json
```

The median level is an observation-derived estimate, not proof that an optimizer
stopped learning. Sparse blanks remain absent readings; only explicit invalid
values break a segment. Intentional holds, curriculum changes and objectives
that should increase require interpretation outside this detector.
