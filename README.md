# trainspotter

**Spot what went wrong in a training run before you burn another GPU-day.**
Automatic diagnosis of loss curves and training logs.

[![PyPI](https://img.shields.io/pypi/v/trainspotter)](https://pypi.org/project/trainspotter/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![Live demo](https://img.shields.io/badge/live%20demo-sample%20reports-4fc3f7)](https://antonsoo.github.io/trainspotter/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-workbench-ffd21e)](https://huggingface.co/spaces/antonsoloviev/trainspotter)

Every ML engineer has stared at a TensorBoard or W&B chart trying to answer
"is this run okay?" -- and caught the answer late: a spike that should have
triggered a restart three hours ago, an eval curve that's been overfitting
since epoch 4, a learning-rate schedule that never actually warmed up. The
signal was in the log the whole time. trainspotter reads the log a run
already writes and turns eyeballing into a report: which pathologies were
detected, where (step ranges), the evidence behind each, and the usual
fixes. It runs post-hoc on a finished log, live via `watch` on a growing
one, or as a CI gate that fails the build.

**Logging less often must not look like slower training.** In a new controlled
offline W&B capture, the previous checkout reported a **4.95x slowdown** when
the logger changed from every 4 updates to every 20. The source checkout now uses
the recorded `trainer/global_step` and reports **0.991x**. A deliberately delayed
run still reports **2.90x**. When only `_step` is available, the report identifies
logging events and leaves per-training-step timing unavailable. Select a custom
counter with `--step-key`. [Captured records, commands and limits](docs/wandb-axes.md).
These fixes are available from source and are **not yet in the PyPI release**.

The current source also corrects [loss and gradient evidence](docs/diagnostic-evidence.md):
a late low-loss reading does not become "early" because logging is sparse,
constant gradient norms do not establish clipping, and short histories keep
their observed NaN/Inf norms. The retained PyTorch captures include clipped
and unclipped runs with identical pre-clip norm histories.

**NaN/Inf gaps must not invent a loss trajectory.** The current source keeps
spikes, setbacks and terminal-rise comparisons inside finite segments. A
[controlled PyTorch capture](studies/finite-segments/README.md) previously
produced a 30-step setback and an isolated spike by joining observations across
two logging gaps. Those unsupported claims are gone; the non-finite training
loss still fails the error gate. The alert names the observed loss, since a
NaN logging reduction alone does not prove that optimizer updates diverged.
The source also [bounds possible plateaus by the observed loss level](studies/plateau-evidence/README.md):
a recovered fourfold rise cannot be folded into one "barely moved" interval.
The report distinguishes the smoothed level from raw fluctuations, retains
noise estimates, and cannot use a future learning rate to hide an earlier stall.
Known-step nonnumeric cells now preserve those boundaries too: dropping
`LOGGER_ERROR` from a loss column could invent a 30-step setback and hide a real
later event. The [unreadable-value repair](studies/unreadable-values/README.md)
keeps the gaps visible in CLI/JSON/HTML, preserves actual NaN/Inf alerts, and adds
an offline download of unreadable coordinates.
These corrections are available from source; no PyPI release was made for them.

## Calibration study on 3,145 real runs

Many people upload `Trainer`'s checkpoint directories to the Hugging Face Hub,
`trainer_state.json` included, so thousands of real training logs are public.
[**The census**](studies/hub-census/README.md) collects 3,145 of them, runs
trainspotter over each, and compares its findings with how the run ended.

The original calibration study below predates the later reader, scheduler, and
[evaluation-evidence repairs](docs/evaluation-evidence.md). Its numbers describe
that retained study snapshot, not a fresh accuracy estimate for the current checkout.

The released detectors (0.2.6) did badly at this: they raised a warning or an
error on 75% of the runs that ended at their best loss, and 12% of their errors
were on runs that ended badly. They had been tuned on smooth generated curves,
and a real per-step loss is not smooth. The unreleased detectors in that study were reworked against the corpus, and then scored once on
1,032 further logs from accounts the corpus does not contain:

| | 0.2.6 | Calibration snapshot | Held out: 0.2.6 | Calibration snapshot |
|---|---:|---:|---:|---:|
| Runs that ended at their best loss and were given a warning or an error | 74.6% | 5.7% | 73.0% | 6.9% |
| ... and were given an error | 20.6% | 0.0% | 18.4% | 0.0% |
| Runs that ended badly and were given a warning or an error | 96.1% | 89.9% | 98% | 95% |
| Findings per run | 16.7 | 0.9 | 13.4 | 0.9 |

The first two columns are the 3,145 runs the detectors were tuned on. The zero
is mostly by construction: an error can now come only from the end-of-run
quantities that define a bad ending. What the quieter detectors miss was
measured by damaging 300 held-out runs by known amounts: sustained rises, late
divergence and NaNs are caught 99% of the time or more. A short burst was
missed three times in five at any height, which turned out to be a gap between
two detectors; with it closed, a burst as high as the run's progress or higher
is caught 92% to 98% of the time, and the runs that ended well get the same
warnings as before. [The study](studies/hub-census/README.md#what-it-no-longer-catches)
has the table.

![The trainspotter report for a public fine-tuning run from the Hub. The training loss falls from 2.7 to about 0.5, spikes to 1.9 at step 3,150, then rises to about 1.4 between steps 14,100 and 27,050 before returning; that stretch is shaded. Two warnings are listed: the spike, at 82% of the run's progress, and the setback, 12,950 steps or 33% of the run.](docs/assets/report-public-run.png)

*A real run from the corpus (the log itself is not redistributed). 0.2.6 gave
it an error and three warnings, each for a single noisy point. This checkout
reports the one spike that undid most of the run's progress for a step, and the
setback: a third of the run spent above a loss level it had already reached.*

## Report

The [sample gallery](https://antonsoo.github.io/trainspotter/) includes a
[sparse log of a late loss drop](https://antonsoo.github.io/trainspotter/loss-cadence.html),
[constant gradients without clipping](https://antonsoo.github.io/trainspotter/gradient-concentration.html),
[W&B run with changing logging cadence](https://antonsoo.github.io/trainspotter/wandb-cadence.html),
[timing without a training counter](https://antonsoo.github.io/trainspotter/wandb-logging-only.html),
[a measured slowdown](https://antonsoo.github.io/trainspotter/wandb-slowdown.html),
[scheduler reset below its old peak](https://antonsoo.github.io/trainspotter/scheduler-resume.html),
[disjoint training and evaluation intervals](https://antonsoo.github.io/trainspotter/evaluation-overlap.html),
and a [frozen model evaluated on changing cohorts](https://antonsoo.github.io/trainspotter/evaluation-cohorts.html).
Each report explains the controlled experiment and embeds its exact input log
for download, including when opened from disk. These reports use the source
checkout; the detector repairs are not yet in the PyPI release.

![Top of the trainspotter HTML report on the bundled divergence example: a header with the run's source and step range, an error/warning/info summary with a health-strip timeline, and the combined train/loss + eval/loss chart -- 320 steps of ordinary cosine-schedule training, then a labeled spike as the run breaks -- followed by the findings log, the non-finite training loss first.](docs/assets/html-report-divergence.png)

*(top of the report -- [full report, all 5 charts + all 7 findings](docs/assets/html-report-divergence-full.png))*

![Terminal output of `trainspotter analyze divergence.trainer_state.json --fail-on error`, showing all 7 findings ordered by severity -- one ERROR (the training loss became Inf), two WARNINGs (the evaluation-loss and training-loss spikes just before it), then four INFO findings (the evaluation rise attributed to the spike, the LR discontinuity that caused it, the gradient-norm explosion and the non-finite gradient norm) -- each with step range, detector name, message, and a one-line fix suggestion.](docs/assets/terminal-divergence.png)

Browse every bundled example's full HTML report at
**[antonsoo.github.io/trainspotter](https://antonsoo.github.io/trainspotter/)**
(built from the source checkout by `scripts/build_site.py`). The two images above are real output from
`examples/divergence.trainer_state.json`, a run built to fail (see
[Demo data](#real-demo-data) for how).

## Quickstart

```bash
pip install trainspotter
git clone --depth 1 https://github.com/antonsoo/trainspotter && cd trainspotter
trainspotter analyze examples/divergence.trainer_state.json
```

Those commands install the PyPI release. To use the current reader, detector,
and long-history improvements, install the cloned checkout:

```bash
python -m pip install .
```

Or point it at your own log:

```bash
trainspotter analyze path/to/trainer_state.json --output html --out report.html
```

## Features

- **Evaluation evidence with limits:** smooth improvement is not labelled noise;
  later training cannot explain an earlier evaluation interval. Reports retain
  exact overlapping spans and treat non-finite readings as gaps.
  [Captured model runs, independent score checks, and public-log audit](docs/evaluation-evidence.md).

- **Five readers**, auto-detected from the file: Hugging Face
  `trainer_state.json`, generic CSV/JSONL, Weights & Biases CSVs (a run's
  history, or a chart's export), PyTorch Lightning `CSVLogger`
  `metrics.csv`, and TensorBoard logdirs (`pip install
  trainspotter[tensorboard]`).
- **Reads logs the way tools really write them.** The names
  `transformers.Trainer` sends to W&B and TensorBoard (`train/learning_rate`,
  `train/grad_norm`), Lightning's `train_loss_step` and `lr-AdamW`, Keras's
  `train/` and `validation/` directories, `add_scalars`' directory per
  curve, a run resumed from a checkpoint, and a W&B chart export or a
  logdir that holds several runs (one is analysed, the others are listed;
  `--run` picks). See [What it reads](#what-it-reads).
- **Ten detectors** -- see the [table below](#detectors) -- each with a
  documented algorithm, tunable thresholds, and stated false-positive modes.
  A finding says what the problem cost the run (a share of its progress, a
  share of its steps), and its severity follows that cost.
- **Three report formats**: a color-coded terminal report, JSON (stable
  schema, for tooling), and a self-contained HTML report (inline SVG
  charts, no CDN, no JavaScript -- opens from disk, works offline forever).
- **Learning-rate reversals (source checkout)**: finds a renewed rise after
  decay, even below the original peak, with exact peak/low/confirmation steps.
  A missing-scheduler-state resume and a cosine-horizon overrun previously
  went unreported. [Captured optimizer runs and limits](docs/learning-rate-rebounds.md)
  show both, plus correct-resume and intended-cycle controls; observations stay informational.
  [Unusable LR readings](studies/lr-gaps/README.md) now bound adjacency without
  hiding supported changes elsewhere or inventing jumps through a logging gap.
- **CI gating**: `--fail-on warning|error` exits non-zero if any finding at
  or above that severity was found, or the input could only be partially read.
  The source checkout adds `--require-timing`: a speed gate can also reject an
  unavailable timing comparison. Reports expose the timing source, exact windows
  and unavailable reasons. See [the captured logging experiment](docs/timing-evidence.md).
- **Live tailing**: `trainspotter watch <file>` re-reads a growing log on
  an interval and prints only newly-appeared findings.
- **numpy is the only required dependency.** No pandas, no plotting
  library, no web framework. (`tensorboard` support is an optional extra
  for TensorBoard's event-file decoder; it does not require pandas or TensorFlow.)

## Usage

```bash
# Terminal report (default), with a CI-style exit code
trainspotter analyze run/trainer_state.json --fail-on error

# Machine-readable JSON
trainspotter analyze run/trainer_state.json --output json > findings.json

# Self-contained HTML report
trainspotter analyze run/trainer_state.json --output html --out report.html

# Force a reader instead of auto-detecting from the extension
trainspotter analyze run/metrics.csv --format lightning

# A TensorBoard logdir, or a W&B chart export, with several runs in it
trainspotter analyze runs/ --run lr-1e-3

# Tail a log that's still being written
trainspotter watch run/trainer_state.json --interval 5 --fail-on error
```

Real terminal output (from `examples/overfitting.trainer_state.json`, one
finding shown):

```
[WARN ] steps 125-799      overfitting Possible overfitting
  Lowest logged eval/loss in this finite segment was 0.526 at step 125; the last value is 0.5554
  (5.6%, slope diagnostic p<0.0001). Within their shared interval, train/loss trends down (675
  readings, steps 125-799) while eval/loss trends up (136 readings, steps 125-799). This is
  consistent with overfitting, but does not establish its cause.
  metric: eval/loss
  fix: If a checkpoint was saved near step 125, compare it with the last saved checkpoint on the same evaluation examples.
```

### What it reads

| Source | What is in it | How it is read |
|---|---|---|
| `trainer_state.json` | `log_history` of one `transformers.Trainer` run | Every entry with a `step`; the closing summary entry (`train_runtime`, the run's average `train_loss`) is metadata, not a point |
| W&B run history CSV | `_step`, `_runtime`, one column per logged key | One recognized training counter or `--step-key`; otherwise `_step` remains a logging axis. Other `_`-prefixed bookkeeping columns are skipped |
| W&B chart export (`wandb_export_*.csv`) | `Step`, then `<run> - <metric>`, `...__MIN`, `...__MAX` for every run drawn | One run; default `Step` is a logging axis. The MIN/MAX band columns are not metrics |
| W&B runs table | One row per run (`Name`, `State`, `Runtime`, config...) | Refused: its rows are different runs, not steps |
| Lightning `metrics.csv` | `step`, `epoch`, sparse metric columns | Blank cells skipped; `train_loss_step` is the loss curve, `train_loss_epoch` keeps its name |
| TensorBoard logdir | Legacy scalar summaries and tensor-valued scalars, in one directory or a tree | Split directories (`train/`, `validation/`) and `add_scalars` directories (`Loss_train/`) are parts of one run; other subdirectories are runs. Modern event streams use explicit restart markers, never wall-clock sorting or an inferred restart from an out-of-order reading |
| Any CSV / JSONL | A `step` column and numeric columns | An explicit step must be an exact integer. With no step column anywhere in the source, row indices are used and identified in the report |

Names are mapped onto the vocabulary the detectors look for (`train/loss`,
`eval/loss`, `lr`, `grad_norm`, `step_time`, `throughput`, `epoch`), one
logged metric per name: of `lr-AdamW/pg1` and `lr-AdamW/pg2`, the first is
`lr` and the second keeps its own name. A name that only looks alike
(`val/box_loss`, `test/loss`) is left alone.

Reports expose the original name behind every metric. Names such as
`eval_accuracy` and `eval/accuracy` remain separate curves even though they
would normally normalize to the same spelling. TensorBoard writer directories
that claim the same metric also remain separate.

An invalid, missing, fractional, or out-of-range value in an **explicit step
column** is skipped and counted; it never turns into a row number. A duplicate
CSV header is rejected; a JSONL object with duplicate keys is skipped. Nonblank,
nonnumeric cells in recognized or otherwise numeric metric columns are reported.
Sparse blanks and nulls remain ordinary missing observations. NaN/Inf metric
values are retained as diagnostic evidence.

For CSV, JSONL, Lightning and W&B, `--step-key 'optimizer/updates'` selects an
exact column name. W&B histories automatically use a single `trainer/global_step`,
`train/global_step`, `global_step` or separate `step` column. Several candidates
require an explicit choice. Missing or invalid values in that counter are never
filled from `_step`. See [axis selection](docs/wandb-axes.md) for chart exports and
the case where the logger explicitly received `step=optimizer_step`.

**Repeated steps are evidence, not extra training progress.** The source checkout
counts equal readings of the same metric/step once for analysis. Conflicting
values exclude that metric from trajectory checks; other metrics remain usable.
Different wall times at the same step stay unknown. Reports retain the original
readings, show analysis counts, and make unresolved conflicts fail a configured
CI gate with exit 2. [Inspect the repeated-step examples](docs/repeated-steps.md).
This behavior is unreleased and is not in the published 0.2.6 package.

A source with several runs is analysed one run at a time: the run with the
most points, unless `--run` names another (its name, or the start of it).
Real output, for two of the bundled example runs written as one TensorBoard
logdir:

```
$ trainspotter analyze runs
source   runs  [tensorboard]
run      lr-3e-4  1 of 2 in this source; --run picks another: lr-1e-3
metrics  eval/loss, grad_norm, lr, train/loss
findings 0 error  0 warning  0 info

No pathologies detected. Only the available metrics and detector checks were evaluated.

$ trainspotter analyze runs --run lr-1e-3
source   runs  [tensorboard]
run      lr-1e-3  1 of 2 in this source; --run picks another: lr-3e-4
metrics  eval/loss, grad_norm, lr, train/loss
findings 1 error  2 warning  4 info
...
```

### CI usage

```yaml
- name: Diagnose the training run
  run: trainspotter analyze runs/latest/trainer_state.json --fail-on error
  # exits 1 if any error-severity pathology was detected
```

| Exit | Meaning |
|---|---|
| `0` | Report produced. With `--fail-on`, input was fully read and no finding met the configured severity |
| `1` | A finding met the configured severity |
| `2` | Input/output error, no numeric metrics, missing optional dependency, or incomplete input with `--fail-on` |

For a performance gate, the source checkout supports:

```bash
trainspotter analyze run/metrics.csv --fail-on warning --require-timing
```

`--require-timing` returns `2` for incomplete input or unavailable timing, after
writing the report. With usable timing, `--fail-on` controls finding severity as
usual. The same option on `watch` stops on an unavailable comparison, including
when a growing log has too few readings. Without this option, loss-only logs
remain useful and do not fail solely because timing is unavailable.

Wall-clock timing requires explicit training-step counts. A CSV containing
only loss and timestamps uses row indices; a change in its logging interval
cannot establish a training slowdown. Direct `step_time` or throughput readings
remain usable without step counts because their units are already recorded.
The [timing guide](docs/timing-evidence.md) shows the measured counterexample,
the report's source/window evidence, and the limits. These additions are not
in the published 0.2.6 package.

A partial report is still written before a configured gate returns `2`. Its JSON
contains `run.input_complete: false` and `summary.input_complete: false`, plus
skipped row/value counts and unread event-file bytes. Without a gate, partial
reports return `0` and show the limitation. `input_complete` concerns reading the
supported data; it does **not** mean training finished or every pathology was checked.

`watch --once` performs one read, including when the file is missing or malformed,
and returns `2` if it cannot read a run. Continuous watching retries transient read
errors, holds its initial run selection, and re-evaluates findings when prior data
changes. An extending finding is printed once; a severity change is printed again.
Report files are replaced atomically, and `--out` cannot overwrite the source log.

The [input integrity contract](docs/input-integrity.md) documents restart semantics,
the supported step range, provenance fields, and the verification limits.

## Detectors

Every detector's full algorithm and false-positive list lives in its
module's docstring (`src/trainspotter/detectors/*.py`); this table is the
summary.

| Detector | Detects | How | Stated false-positive modes |
|---|---|---|---|
| `spikes` | A point far above the loss level around it | More than 6 robust standard deviations out (scored on the log of the loss, against the larger of the run-wide and trailing-window scatter) **and** at least half the run's progress above the level (a warning), or a tenth (one note counting them) | Deliberate LR restarts/curriculum jumps; too few points for a stable scale |
| `setbacks` | The loss level rose, stayed up, and came back | The smoothed level sits a quarter of the run's progress above a level it had already reached; reported with the steps spent there, a warning from 5% of the run. One that starts in the first tenth of the run is an "early rise". A rise shorter than the smoothing window is a "loss burst" (a warning) when the level stood half the run's progress above its earlier best and more than 6 times the scatter of single points | An epoch boundary or a planned restart; a rise shorter than half the smoothing window (4% of the run, at least five points) does not move the level and is left to `spikes` |
| `divergence` | A non-finite loss, or a rise the run never recovered from | Any NaN/Inf (an error in the training loss, a warning in an evaluation metric); or the log ends inside a rise with a quarter (warning) or half (error) of the run's progress given back; or the loss only ever rose | A run stopped during a temporary rise; a metric meant to increase |
| `plateau` | Possible stalled progress within a finite segment | OLS and endpoint filters plus a bound on the median level, including the whole merged interval. The bound allows observed noise; reports retain both level and raw ranges. Suppress near the segment's floor or an already-observed LR decay. [Algorithm and controls](studies/plateau-evidence/README.md) | Noisy or correlated losses, intentional holds, and changes shorter than the smoothing window; no proof of convergence |
| `overfitting` | Evaluation loss rising after its lowest logged value | Final finite segment only; positive slope diagnostic and ≥ 3% rise, or a large endpoint rise. Notes below 10%, warnings from 10% (5% with a strong slope), errors from 50%; zero baselines give a note without a percentage. Possible overfitting requires opposing slopes within a shared training/evaluation interval | Selected minimum and correlated readings prevent calibrated inference; evaluation-data changes can produce the same pattern |
| `lr_schedule` | No warmup, rises after decay, discontinuities (all notes) | First value vs. peak; time-ordered decline and rise exceeding 1% of their peak scale; a jump exceeding 20% of the LR's total range that isn't part of a steady ramp or initial warmup. Rebounds retain exact observations; the same upward jump is not reported twice | Cyclic/warm-restart schedules also rise after decay; intent is unknown. Step-decay drops can be reported |
| `grad_norm` | Non-finite norms, explosions, and concentration (notes) | One NaN/Inf is reported even in a short or ambiguous history. Finite segments: unusual readings at least 10x their rolling median. A finite, nonnegative trailing window: fraction within 0.5% of its own maximum | A constant gradient and post-clip logging can look the same. Norms alone do not establish clipping, its threshold, or its frequency; [recorded counterexamples](docs/diagnostic-evidence.md) |
| `throughput` | Step-time / throughput regressions | Median step time, early-run baseline window vs. recent window, flags ≥ 1.4× | Checkpoint/eval steps inflate single points; early-run kernel/dataloader warmup can pollute the baseline |
| `eval_noise` | Repeated evaluation fluctuations (a note) | At least two direction reversals and median absolute adjacent change ≥ 15% of the observed range, within a finite segment of ≥ 6 readings | Cadence-dependent description, not a noise estimate. Changing checkpoints or evaluation data can both cause swings |
| `loss_floor` | Low train loss early in the logged step span (heuristic) | Loss <= 0.05 absolute **and** <= 5% of its finite, positive first reading, within the first 10% of the observed training-step interval. Requires a training-step axis and finite, nonnegative readings in that window | An easy task, a small-scale loss, or a resumed checkpoint can look the same. Sparse logging does not make a late observation early; [same-run comparison](docs/diagnostic-evidence.md) |

"Progress" is how far the loss level fell from where the run started to the
best level it reached. It is the yardstick for every "how bad was that"
judgement, because differences in a loss carry meaning and ratios do not: a
cross-entropy going from 0.10 to 0.15 moved 0.05 nats, the same as one going
from 2.00 to 2.05.

### Trust the healthy case

A tool that flags an unremarkable run gets ignored, so being quiet on healthy
runs comes first. Both sides are measured: of 537 held-out public runs in the
[census](studies/hub-census/README.md) that ended at their best loss, this
checkout gives none an error and 6.9% a warning (the released 0.2.6 gave 18.4%
an error and 73.0% a warning or an error), and of the same kind of run with a
fault of known size put into it, it reports the sustained ones and, once they
are as high as the run's progress, 92% or more of the short bursts.
`tests/test_field_calibration.py` rebuilds the shapes those logs take (the
heavy-tailed scatter of a per-step loss, a warmup logged every few steps, a
level that climbs for a tenth of the run and returns) and pins what each
detector says about them; `tests/test_examples_regression.py` asserts the
bundled healthy baseline stays silent and the other four examples still flag
what they were built to show. Findings are ordered errors-first, then
warnings, then notes (chronological within a level).

## How it works

**Data model.** Every reader converges on one shape
(`trainspotter.model.Run`): a dict of named metric series, each a list of
`(step, value, wall_time)` points. Detectors and reports only ever see
this shape -- they have no idea whether the log came from `transformers`,
a CSV, or TensorBoard. Readers normalize common spellings (`loss` /
`train_loss` / `Loss/train` / `train_loss_step` become `train/loss`;
`learning_rate` / `train/learning_rate` / `lr-AdamW` become `lr`; see
[What it reads](#what-it-reads)) but pass anything unrecognized straight
through, so a custom metric like `eval/bleu` still reaches the detectors
under its own name.

**Levels, not points.** A logged training loss is one mini-batch's average:
single points scatter widely, with a heavy upper tail, around a level that
moves slowly. The detectors that judge whether something hurt the run
(`spikes`, `setbacks`, `divergence`) first reduce the series to its levels
(`detectors/base.py`: a centred rolling median, where the run started, the
best level it reached, where it ended) and measure everything against the
run's progress between start and best. A point is a spike only if it is both
unusual and large on that scale.

**Robust statistics, not raw thresholds.** Scatter is estimated from medians
and median absolute deviations (and from successive differences in short
series), not a mean/stddev: those are dragged around by the very outlier
being judged. Plateau and overfitting use an ordinary-least-squares slope with
a normal-approximation t-test (`math.erf`-based, no `scipy` dependency)
rather than eyeballing "did it go up or down."

**Long histories.** The source checkout batches median calculations and reuses
local regression moments across overlapping plateau windows. Every observation
still contributes to diagnosis; numerically uncertain slope decisions use a
direct fit. [Algorithms, scaling measurements, and a million-observation
report](docs/long-histories.md) explain the memory and accuracy boundaries.

**Charts don't let one spike flatten the rest of the curve.** The HTML
report's axis logic (`report/html_report.py`) picks tick spacing with
Heckbert's "nice numbers" algorithm (`Graphics Gems`, 1990 -- 1/2/5x10^n
steps, not whatever an even split of the range happens to produce) and
floors the axis at 0 for the non-negative metrics trainspotter charts
(loss, LR, grad_norm, accuracy, step time), instead of padding below zero.
If the max is more than 1.5x the 99th percentile, the axis caps at that
instead of stretching to fit one outlier -- the point is still drawn,
clamped to the top edge with a small triangle and its real value labeled
next to it, not hidden.

**Every finding is honest about its own limits.** Each detector's
docstring states its false-positive modes in plain language (see the table
above, or the source for the full version) -- this isn't boilerplate, it's
meant to be read before you act on a finding. `loss_floor` is explicitly
labeled a heuristic, not a statistical test, because there's no
distribution-free way to know a loss is "too low" without knowing the task.

## Real demo data

Nothing in `examples/` is fabricated. All five logs were produced by
`examples/generate_examples.py`, which trains a small hand-written numpy
MLP (64 → 32 ReLU → 10, softmax cross-entropy, every gradient computed and
applied by hand -- no autodiff framework) on scikit-learn's `load_digits`
dataset (1797 real 8×8 handwritten-digit images, bundled with
scikit-learn, no download). It's deterministic (seeded) and takes about
three seconds to regenerate on this machine:

```bash
uv sync --extra examples   # or: pip install scikit-learn
uv run python examples/generate_examples.py
```

| File | What it is | How the pathology was actually induced |
|---|---|---|
| `baseline` | A normal, unremarkable run | Full dataset, weight decay, cosine LR with warmup. **On this run, trainspotter reports no warnings** -- see [Trust the healthy case](#trust-the-healthy-case) for why that's asserted by a test, not just eyeballed |
| `divergence` | A genuine crash after real training | 320 steps (80% of the run) of ordinary warmup + cosine-decay training -- loss ~2.8 -> ~0.15, eval accuracy up to ~96% -- then a simulated incident at step 320: the LR schedule jumps to 30 **and** the loss function switches to a version with the classic missing-max-subtraction softmax bug. Plain high LR alone was tried first and *didn't* produce real divergence -- see the note in `divergence_run()`'s docstring for why (softmax cross-entropy's gradient is bounded by construction) |
| `overfitting` | Genuine overfitting | Only 4 training examples per class (40 total), no regularization, 800 steps -- the model memorizes the training set while held-out eval loss turns upward after step 125 |
| `missing_warmup` | No LR ramp-up | LR set to a constant 0.5 from step 0, no warmup phase at all |
| `throughput_drop` | A real, measured slowdown | Steps 150-259 insert an actual `time.sleep(0.02)` per step (simulating e.g. I/O contention) -- `step_time` in the log is genuinely measured wall-clock time, not a fabricated number |

Each file is committed in both HF `trainer_state.json` format
(`<name>.trainer_state.json`) and generic CSV (`<name>.csv`), written from
the same in-memory log so the two are guaranteed consistent.

### A real `transformers.Trainer` run

`examples/transformers_tinygpt2.trainer_state.json` is a genuine
`trainer_state.json` written by `transformers.Trainer` itself (not the
numpy MLP above): a real 2-layer, 64-dim GPT-2 (`GPT2LMHeadModel` from a
from-scratch `GPT2Config`, ~330K parameters), trained for 6 epochs / 726
steps on CPU on a small hand-written toy corpus (short sentences about
training, tokenized with the real `gpt2` tokenizer), in about 90 seconds
on this machine. Regenerate it with
`examples/generate_transformers_example.py` (needs `torch` + `transformers`
in a separate environment -- see the script's docstring for the exact
install commands; they are *not* project dependencies).

Running trainspotter against it caught a real bug in the HF reader during
development: `Trainer.train()` appends one extra `log_history` entry after
training ends -- a run summary with `train_runtime`, `train_samples_per_second`,
and a `train_loss` key that is the *average* loss over the whole run, not a
per-step reading. The reader's first version merged that average straight
into the `train/loss` series (since `train_loss` is one of the recognized
spellings of `loss`), which showed up as a fake spike on the run's last
step. The fix -- detect that entry by its unique `train_runtime` key and
route it to run metadata instead of a metric point -- is
`src/trainspotter/readers/hf.py`'s `_SUMMARY_MARKER_KEY`, and
`tests/test_readers.py::test_hf_trainer_state_excludes_the_final_run_summary_entry`
pins it. This is exactly the kind of gap a hand-written fixture can miss
and a real framework's output finds immediately -- the reason this example
was worth the extra `torch` install.

It also exposed a false positive, now fixed: the first version reported an
`lr_schedule` "LR discontinuity" at steps 5-10, because `logging_steps=5`
logs the 20-step warmup so coarsely that each logged interval covers 25% of
the run's LR range. The discontinuity check now reports a jump only when it
stands out from the logged intervals around it (a neighbor moving the same
way at no less than half the per-step rate makes it one leg of a steady
ramp), and says how many steps the interval spans. On this run trainspotter
reports nothing, which
`tests/test_detector_lr_schedule.py::test_the_real_transformers_run_has_no_lr_finding`
pins.

## Accuracy and limitations

- The detectors that carry a warning or an error were calibrated on 3,145
  public `Trainer` logs ([the census](studies/hub-census/README.md)), against
  how each run ended. Nobody labelled those runs by hand, they are mostly
  small fine-tunes, and "ended badly" is computed from the same loss values
  the detectors read; the study says what that does and does not establish.
  Every threshold is still a default. `plateau`, `throughput`, `eval_noise`
  and `loss_floor` were not recalibrated: their thresholds are the ones tuned
  on the bundled examples.
- **Readers are tested against real files**, not remembered schemas: every
  reader in `tests/test_readers.py` is exercised against a file this repo's
  test suite writes to disk in the documented format. The TensorBoard
  reader is tested against real event files written by TensorBoard's own
  `EventFileWriter`. `tests/test_real_layouts.py` writes the bundled example
  runs the way other tools would have logged them (a W&B history and chart
  export of a `transformers` run, four TensorBoard layouts, a Lightning
  `metrics.csv`) and requires the findings the original gives. The W&B
  chart export's column layout was checked against exports committed to
  public repositories; it is not written by W&B in these tests.
- **One run at a time.** From a source with several runs, one is analysed
  and `--fail-on` gates that run only. A run is told from a split of a run
  by its directory name (`train`, `validation`, `val`, `eval`, `test`), so
  a run literally named `train` next to other runs is read as a split.
  In this source checkout, TensorBoard read failures are also scoped to that
  selection; damaged sibling runs remain in JSON directory evidence. Unread
  content with ambiguous directory ownership still makes the possible owner
  incomplete. [Controlled gate replay](studies/tensorboard-scope/README.md).
- **Restart evidence is explicit in modern TensorBoard streams.** Every
  `SessionLog.START`, including the first observed marker, invalidates prior
  points at or beyond that step in the same writer directory. This follows
  TensorBoard's legacy scalar accumulator's convention, applied equally to
  scalar and tensor storage; other TensorBoard loaders can differ. Without a
  marker, out-of-order and duplicate-step readings are retained. Older,
  unversioned/v1 streams use the legacy per-tag rewind heuristic. Event files
  are read in filename order, not wall-clock order; do not rename them to
  imply a different order. See the contract for oracle details.
  Retention does not make conflicting values a usable trajectory: repeated-step
  analysis and evidence are described in [the integrity contract](docs/input-integrity.md).
- **Detector tests use synthetic signals with known ground truth**: a spike
  planted at step *k* must be found at step *k*; a clean curve must produce
  zero findings (`tests/test_detector_*.py`, `tests/test_field_calibration.py`).
  A regression test (`tests/test_examples_regression.py`) runs the whole
  detector set against the five bundled example logs. The public logs
  themselves are not in the repository; the census is how the detectors meet
  them.
- No detector here is a substitute for understanding your training run.
  They're heuristics built to reduce how often you have to stare at a
  chart by eye, not a certified diagnosis -- read the false-positive modes
  in the [detector table](#detectors) before trusting a finding blindly.
- `trainspotter watch` re-reads the whole file on every poll (default
  every 2s) rather than tailing new bytes; fine for the log sizes this
  tool targets (thousands to tens of thousands of steps), not designed for
  gigabyte-scale logs.
- Timing comparisons require finite positive readings in both comparison
  windows. Missing or invalid readings do not become a zero or move the window
  into earlier data. Inverted throughput is seconds per reported unit (sample
  or token), not seconds per training step.
- The OLS diagnostics (`plateau`, `overfitting`) use a normal approximation
  to the t-distribution. Small windows, correlated readings, and selecting
  the minimum from the same history prevent calibrated probabilities of a
  training fault. `eval_noise` describes fluctuations; aggregate histories
  do not measure checkpoint uncertainty. See [evaluation evidence](docs/evaluation-evidence.md).

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check src tests examples
uv run mypy src
uv build
# Optional browser checks for the generated, self-contained reports:
npm ci
npm run test:browser
```

The browser checks require installed Playwright Chromium and Firefox browsers
(`npx playwright install chromium firefox` on a new machine). They exercise the
actual generated reports, including keyboard access to metric provenance and
scrollable charts at 320px, axe accessibility, CSP, and absence of network requests.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the detector-contribution
pattern.

## Contributing

Issues and PRs are welcome -- see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) &copy; 2026 Anton Soloviev

---

<sub>Part of [Officina](https://antonsoo.github.io/officina/), a set of small open-source tools by [Anton Soloviev](https://github.com/antonsoo).</sub>
