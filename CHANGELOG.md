# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## Unreleased

- Check TensorBoard record lengths against available local file bytes before
  decoding. Damaged length headers no longer cause an enormous allocation and
  lose the preceding scalar report. Retain unread bytes and incomplete-input
  gates; the SDK still checks both CRCs. Files growing during a pass are reread
  on the next load. [Reproduction and limits](studies/record-lengths/README.md).

- Protect every discovered TensorBoard event file from report output, including
  unselected runs and path aliases. Reject new event-like report filenames inside
  log directories so later reads remain valid. Ordinary in-directory reports
  still replace atomically. [Reproduction and limits](studies/report-inputs/README.md).

- Reject ambiguous W&B chart run/metric labels instead of silently splitting at
  the last separator. Add `--chart-single-run NAME` for a declared single-run
  chart, preserving complete metric names, report provenance and watch reloads.
  Ordinary chart run selection is unchanged. Separator-containing run names also
  require an explicit declaration or a run-history export.
  [Behavior and limits](docs/wandb-chart-identity.md).

- Keep supported LR discontinuities visible when other readings are unreadable,
  non-finite or negative. Peak/range calculations use finite nonnegative values;
  ramp comparisons and phase detection stay within contiguous usable segments.
  Negative values cannot create a spurious jump, and a later usable reading
  cannot replace an unknown beginning in the missing-warmup check. Reports retain
  the usable range and segment scope. [Controlled optimizer capture and limits](studies/lr-gaps/README.md).

- Preserve known-coordinate unreadable metric values as trajectory boundaries in
  CSV, JSONL, Hugging Face, W&B, Lightning and TensorBoard scalar inputs. Invalid
  cells no longer invent rises, spikes or recovery durations across gaps, or hide
  a valid local event after one. Keep sparse cells absent and actual NaN/Inf
  observations distinct, including repeated steps and watch rewrites.
- Expose unreadable coordinates in JSON and standalone HTML, with broken chart
  lines, separate markers and a complete offline evidence download. Reading counts
  still count numeric observations; all-unreadable input remains an input error.
  [Reproduction, controls and limits](studies/unreadable-values/README.md).

- Bound possible plateaus by their median loss level and observed local noise,
  including the full union of merged windows. Equal endpoints and a weak linear
  slope no longer make a recovered rise into a flat interval. Loss gaps split
  baselines; LR suppression uses only prior finite, nonnegative evidence.
  Retain raw ranges and noise allowances, and report "Possible plateau".
  [Before/after examples, public logs and noise controls](studies/plateau-evidence/README.md).

- Keep loss spikes, recovered setbacks and terminal-rise baselines within
  contiguous finite observations. NaN/Inf gaps no longer join separate levels
  or manufacture recovery durations; valid events after a gap remain visible
  with explicit segment bounds and duration shares. Name a non-finite training
  loss without asserting optimizer divergence; retain its error severity.
  [Captured PyTorch counterexample and CLI replay](studies/finite-segments/README.md).

- Scope TensorBoard skipped-event and unread-byte counts to the selected run.
  Retain other writers' integrity evidence in JSON, preserve damaged runs with
  zero scalar points, and conservatively attribute unread `add_scalars` writers.
  Impossible record lengths now produce incomplete evidence instead of an
  uncaught overflow. [Before/after CLI replay](studies/tensorboard-scope/README.md).

- Measure `loss_floor`'s early window by the observed training-step span,
  not logging frequency. Require a training axis and valid early baseline.
  Describe gradient concentration without inferring clipping; retain single
  non-finite norms and non-finite evidence in conflicting histories. Invalid
  norms cannot define a ceiling, and gaps separate gradient comparisons.
  [CPU PyTorch counterexamples and exact inputs](docs/diagnostic-evidence.md).

- Reduce repeated work on long loss histories: batch rolling medians/MADs with
  bounded temporary storage and use locally rebased regression moments for
  plateau windows. Keep the existing window definitions and diagnostic rules;
  uncertain numerical decisions use a direct fit. Correct OLS residual precision
  at large offsets. [Algorithms and reproduction](docs/long-histories.md).

- Distinguish W&B logging counters from training steps. Use a single recorded
  training counter when present; keep `_step`/default chart `Step` visibly on a
  logging axis otherwise, with wall-clock per-training-step timing unavailable.
  Add `--step-key` to analyze/watch and `step_key=` to tabular library readers.
  Ambiguous counters require a choice; invalid selected coordinates never fall
  back to logging positions. JSON, HTML and terminal reports expose the axis
  and explicit selections. [Offline SDK capture and verification](docs/wandb-axes.md).

- Add three public reports for scheduler resets, disjoint evaluation/training
  intervals and deliberate evaluation-cohort changes. Every gallery report
  now carries its capture context and exact input log as an offline download.

- Evaluation evidence: `eval_noise` now describes repeated fluctuations, without
  claiming checkpoint uncertainty; monotone curves, single turns, and NaN/Inf
  gaps cannot manufacture noise findings. Evidence fields now describe scaled
  adjacent changes, range, direction counts, and finite observations.
- `overfitting` requires contemporaneous opposing loss trends for its "Possible
  overfitting" title. It retains both observed spans, assesses only the final
  finite evaluation segment, corrects negative-baseline thresholds, and marks
  zero-baseline percentages unavailable. Suggestions no longer assume a saved
  checkpoint exists. [Captures and audit](docs/evaluation-evidence.md).


### Learning-rate reversals

- Repair the check for LR rising after decay: it previously looked for a
  value above the entire run's maximum and could never report a rise. Walk
  the observations in order, retaining the preceding peak, low, first
  confirmation and high of each rebound, including below the original peak.
- Keep reversals informational: intentional cycles and warm restarts can
  produce the same shape. An upward discontinuity already described by a
  rebound is not reported twice. Unusable LR readings break the evidence chain.
- Default `peak_grace_frac` to zero so appended observations cannot hide an
  earlier rebound. Explicit nonzero values still ignore an initial fraction
  of logged points. Ordinary warmup is excluded by requiring a preceding decline.
- Retain eight controlled PyTorch training captures, independent CLI replay,
  correct-resume and intentional-cycle controls, and a public-log audit.
  The audit describes 174 observed reversals across 82 logs, without claiming
  that their schedules were faulty. See `docs/learning-rate-rebounds.md`.

### Timing evidence and gates

- Stop deriving seconds per training step from a reader's row-index fallback.
  A change in logging cadence could previously produce a throughput warning;
  that comparison now stays unavailable. Directly logged step time and
  throughput remain usable without a step column.
- JSON, terminal and offline HTML reports expose the selected timing source,
  baseline/recent windows, units, unusable counts and unavailable reasons.
  Invalid window values remain in place, so a broken clock cannot produce a
  convenient comparison by sliding the window. The JSON `timing` object is an
  additive schema-v1 field and uses the default detector window settings.
- `analyze` and `watch` accept `--require-timing`. It exits 2 for incomplete
  input or an unavailable timing comparison. Pair it with `--fail-on warning`
  to gate measured slowdowns; an analysis report is written before the gate exits.
- Add measured, deliberately paced training captures, an independent standard-library
  clock calculation, and three timing examples to the static report gallery.

### Changed

- **The detectors were recalibrated on 3,145 public `Trainer` logs**
  (`studies/hub-census/`). 0.2.6 raised a warning or an error on 74.6% of the
  runs that ended at their best loss and an error on 20.6%; this version does so
  on 5.7% and 0.0%, and still flags 89.9% of the runs that ended badly. Findings
  per run fall from 16.7 to 0.9. Those are the runs the thresholds were set on,
  and no error on a run that ended well follows from how errors are now defined.
  On 1,032 logs collected afterwards from other accounts: 6.9% and 0.0% (0.2.6:
  73.0% and 18.4%), and 95% of the runs that ended badly. With faults of known
  size injected into 300 held-out runs that ended well, sustained rises, late
  divergence and NaNs are reported 99% of the time or more, and a burst lasting
  2% of the run 92% to 98% of the time once it is as high as the run's
  progress. Expect far fewer findings, and different ones:
  - `spikes`: a point must be large next to the run's progress as well as
    statistically unusual. Scored on the log of the loss against the run-wide
    scatter. A warning from half the run's progress, one note below that. Never
    an error; downward points and a run's initial descent are not spikes.
  - `divergence`: the rising-tail test is replaced by "the log ends inside a rise
    with a quarter (warning) or half (error) of the run's progress given back",
    plus "the loss only ever rose". It no longer reads `eval/loss` for this: a
    rising evaluation loss is `overfitting`'s. A non-finite evaluation metric is
    a warning; a non-finite training loss is still an error.
  - `overfitting`: a note below a 10% rise, a warning from 10% (5% with an
    unmistakable trend), an error from 50%. Runs with three or four evaluations
    are reported when the last is far above the best. Titled "Evaluation loss
    rising" when the training loss was not falling beside it.
  - `grad_norm`: one note per run, counting readings at least 10x the rolling
    median, in place of a warning per reading. A non-finite norm is a note.
  - `lr_schedule`: no warmup is a note. A rise before the learning rate has ever
    fallen is warmup, however coarsely logged, and not a discontinuity.
  - `eval_noise` is a note. `loss_floor` ignores a "loss" that goes negative.
- Agreeing duplicate `epoch` readings, which `Trainer` writes at every evaluated
  step, are no longer listed as repeated-step evidence in the terminal and HTML
  reports. JSON keeps them.
- The HTML report does not chart `*/runtime`, `*/samples_per_second` or
  `*/steps_per_second`, and charts at most twelve further metrics; the rest are
  named under the last chart.

### Added

- `setbacks` detector: a stretch where the loss level sat a quarter of the run's
  progress above a level it had already reached, reported with the steps it
  cost. One that begins in the first tenth of the run is an "early rise".
  A rise shorter than the smoothing window is a "Loss burst" warning when the
  level stood half the run's progress above its earlier best and more than six
  times the scatter of single points; more than five in a run are one "Repeated
  loss bursts" finding. Without it a burst that filled more than half the
  window was too wide for `spikes` and too short to be a setback: three
  injected bursts in five went unreported at any height, and so did a public
  run whose loss stood at 6.8 against a level of 0.83 for 37 logged points.
- `studies/hub-census/`: the collection script, the manifest of the 3,208 files
  read, the census and report scripts, and the results.

### Fixed

- Repeated copies of one step no longer masquerade as a long trajectory (for
  example, a plateau at a single step). Equal values count once in detector
  inputs. A metric with conflicting values at one step is excluded from
  trajectory checks; conflicting timestamps remain unknown. Original readings
  stay available, and unresolved conflicts make configured CI gates exit 2.
- Single-point chart segments now have visible markers.
- Read TensorFlow 2 / Keras tensor-valued scalars as well as legacy TensorBoard
  scalars. Replay restart markers across files and tags in event order; a
  backward clock or out-of-order modern reading no longer erases valid history.
  TensorBoard imports expose unread trailing bytes and count points invalidated
  by restart events. The optional extra uses TensorBoard directly, removing the
  tbparse/pandas dependency chain.
- Validate step coordinates without float rounding or truncation. Invalid
  explicit steps are counted instead of being replaced with row indices.
  Reject duplicate CSV headers and ambiguous step columns; detect duplicate JSON
  keys, incomplete CSV rows, and invalid metric cells. Keep colliding normalized
  metric names separate and prevent input metadata from forging reader counts.
- Detect even a single NaN/Inf loss reading, recognize exactly constant loss as
  a plateau, and avoid fabricated throughput findings from invalid timing windows.
  Throughput reports state their actual time unit; large finite medians do not overflow.
- `watch --once` terminates on missing, empty, or malformed input. Continuous
  watches hold their initial run selection, recover read errors, and reconsider
  rewritten history without repeatedly printing an extending finding. Invalid
  intervals are rejected. Terminal names escape line breaks and bidi overrides.
- `--out` works for terminal reports, refuses the input file (including aliases),
  and preserves an existing report if writing its replacement fails.

### Added

- Repeated-step evidence in JSON, terminal and offline HTML reports, original
  versus analysis point counts, bounded HTML tables and a complete evidence
  download. Controlled captures from TensorBoard's actual event writer document
  replayed records and conflicting readings without a restart marker.
- Input-completeness fields, original metric names, skipped-value counts and
  restart provenance in JSON reports. Incomplete input returns exit `2` when
  `--fail-on` is configured, after writing the available report.
- Visible incomplete-input notices, an expandable metric-source table, and
  keyboard-scrollable phone charts in self-contained HTML reports.
- Regression tests using real event files and an independent TensorBoard scalar
  accumulator, plus repeatable Chromium/Firefox report and accessibility checks.

### Maintenance

- The uv configuration and lock now explicitly prefer stable dependency releases,
  keeping local development and CI consistent.

## [0.2.6] - 2026-10-03

### Security

- A metric or run name holding a terminal escape sequence was printed as it came: the terminal
  report, `watch` and the error messages sent it to the terminal, which obeys it (clears the
  screen, retitles the window, hides the rest of the line). Each control character in text
  from the log is now written as a visible escape: `loss\x1b]0;title\x07`. The JSON and HTML
  reports already escaped them.

## [0.2.5] - 2026-10-03

### Compatibility

- Python 3.13 and 3.14 are tested and declared. CI runs the suite on 3.14
  as well, and the package's classifiers list both versions. The code is
  unchanged: with the newest release of every dependency, the tests pass on
  3.14 and on 3.15's release candidate.

## [0.2.4] - 2026-10-02

### Accessibility

- The HTML report, checked with axe-core (WCAG 2.1 A and AA, and its best-practice rules) in light and dark,
  at desktop and phone widths: no findings now.
  The dim text was 3.1:1; the report has an `h1` and one `main` landmark that
  holds the summary, the charts and the findings.

## [0.2.3] - 2026-10-02

### Security

- The HTML report carries a Content-Security-Policy. It is one file with no
  script in it, and the policy has the browser hold it to that: nothing in it
  may run or be fetched, whatever a metric or a run is named. Names are
  escaped; the policy is for the day one is not. Opened from disk in Chromium
  and Firefox: no violations, and nothing but the timestamp differs from the
  report as it was.

## [0.2.2] - 2026-10-02

Logs as spreadsheets and Windows tools save them. Each of these was checked by
saving `examples/divergence.csv` that way: all now give the report the
original gives.

### Fixed

- A CSV with a byte-order mark (Excel's "CSV UTF-8"). The mark became part of
  the first column's name, so `step` was no longer the step column: it was
  listed as a metric and every finding was reported at its row number (the
  loss spike at step 320 appeared at 353).
- A CSV with semicolons and decimal commas, which is what a spreadsheet saves
  wherever the decimal mark is a comma. No metric was found, and the report
  said "No pathologies detected. Clean run" with exit code 0, for a run that
  diverged. The delimiter (comma, semicolon or tab) is taken from the header,
  and decimal commas are read when the delimiter is not the comma.
- A log in which no numeric metric is found is an error (exit 2) that names
  the columns, not a clean report. With `--fail-on` in CI, a file that could
  not be read used to pass.
- A `trainer_state.json` or JSONL log with a byte-order mark stopped with
  "Unexpected UTF-8 BOM".
- Output written to a pipe or a file is UTF-8. Before 3.15, Python on Windows
  gives a redirected stdout the system's code page, and a metric or run name
  outside it stopped the report with "'charmap' codec can't encode
  characters". (Reproduced on Linux by giving the pipe cp1252 with
  `PYTHONIOENCODING`.)

## [0.2.1] - 2026-10-02

### Fixed

- Python 3.10. `trainspotter` could not be imported there: the HTML report
  used `datetime.UTC`, which Python gained in 3.11, so every command stopped
  at `ImportError: cannot import name 'UTC' from 'datetime'`. This was true
  of every release so far, although each declared `requires-python >= 3.10`.
  The test suite now runs on 3.10 before a release, on its own and with the
  oldest numpy the package allows (1.24.0).
- Array annotations are spelled `NDArray[np.float64]`. A bare `np.ndarray`
  is a complete type only on the numpy releases that need Python 3.11, so
  the strict type check failed on 3.10.

## [0.2.0] - 2026-10-01

### Added

- W&B chart exports (`wandb_export_*.csv`): `Step`, then `<run> - <metric>`
  with `__MIN` and `__MAX` companions for every run drawn in the chart. They
  were read as a generic CSV, so a two-run export of one metric was six
  unrelated metrics of one run. One run is read and the band columns are
  left out.
- `--run NAME` for a source that holds several runs (a W&B chart export, a
  TensorBoard logdir): the run's name, or the start of it. Without it the
  run with the most points is analysed, and every report says which run it
  is about and that there are others (`run.selected_run` and `run.runs` in
  the JSON).
- TensorBoard logdirs as trees. Keras writes `train/` and `validation/`
  directories with the same tags, and `SummaryWriter.add_scalars` a
  directory per curve; both are one run. Other subdirectories are runs.
  Everything under a logdir used to be merged by tag, so Keras's two
  `epoch_loss` curves, or two runs' losses, were one series with two values
  at each step.

### Fixed

- Metric names as tools write them. `transformers.Trainer` reports
  `train/learning_rate` and `train/grad_norm` to W&B and TensorBoard, so the
  LR-schedule and gradient-norm detectors never ran on those logs. Also
  recognized now: `val/loss`, `valid_loss`, `validation/loss`, `Loss/train`
  and `Loss/val`, Lightning's `train_loss_step` and `lr-AdamW`, Keras's
  `epoch_loss` and `epoch_learning_rate`, `lr/pg0`. A canonical name goes
  to one logged metric: `train_samples_per_second` and `samples_per_second`
  in one log were both stored as `throughput`, one series with two values
  at a step.
- A run resumed from a checkpoint leaves a second event file that starts
  below where the first ended. Both were kept, two values per overlapping
  step. The earlier session's points from the restart step on are dropped,
  as TensorBoard does, and the report counts them.
- A W&B runs table (one row per run) was read as a training log: its rows
  became steps, and two runs' learning rates an "LR discontinuity". It is
  refused with the reason; `--format csv` still reads it as it is.
- The throughput detector's wall-clock fallback measured seconds per logged
  point and called it step time, so a log that went from every 10 steps to
  every 50 looked five times slower. It measures seconds per step.
- TensorBoard wall times were never read (the column was not requested), so
  that fallback never ran on TensorBoard logs.

## [0.1.3] - 2026-10-01

### Fixed

- A step that isn't a number a step can be (`nan`, `inf`, `1e400`) crashed
  every reader with `OverflowError`, or stopped the run with "cannot convert
  float NaN to integer". Such a row is skipped and counted.
- Reports never said how many rows were skipped, although the readers counted
  them. The terminal and HTML reports now show the count when it isn't zero
  and the JSON report always carries `run.skipped_rows`, so "no pathologies
  detected" can't quietly rest on half a log.
- On a diverged run, numpy's `RuntimeWarning: invalid value encountered in
  subtract` (and others) was printed above the report that already said the
  run diverged. The detectors are built to take Inf and NaN; the warnings are
  gone.
- A W&B history CSV with a row longer than its header crashed the reader
  (`'NoneType' object has no attribute 'startswith'`).
- A `trainer_state.json` whose `log_history` isn't a list raised `TypeError`;
  it is a one-line error.
- The HTML report failed with `OverflowError` when a metric held values so far
  apart that their difference overflows (`1e308` and `-1e308`). Values beyond
  1e150 are marked like NaN and Inf instead of plotted.
- The HTML report's step range always started at 0, whatever the first logged
  step was.

### Changed

- The HTML report draws at most four points per half-pixel column of a chart
  (the first, lowest, highest and last), which is everything that column can
  show. A metric logged at each of 200,000 steps took 2.6 MB of SVG and now
  takes 93 KB, with a one-step spike still drawn where it happened. Reports
  for shorter logs are unchanged.

### Added

- A seeded fuzz test: 600 generated logs in every supported text format, with
  empty files, repeated and unordered steps, NaN and infinite values and rows
  of the wrong shape, each of which must end in a report or a one-line error,
  with no warning printed.

## [0.1.2] - 2026-10-01

### Added

- Published to PyPI: `pip install trainspotter`. The README's images and links
  are rewritten to absolute URLs at build time so they work on the project
  page.
- `trainspotter --version`.

## [0.1.1] - 2026-09-30

### Fixed

- `lr_schedule` reported a short warmup as an "LR discontinuity" when the log
  was coarse (the real `transformers.Trainer` example: a 20-step warmup
  logged every 5 steps, so each logged interval covers 25% of the LR range).
  A jump is now reported only if it stands out from the logged intervals
  around it, and the message gives the step gap instead of always saying
  "in a single step".
- Slope p-values print as `p<0.0001` instead of `p=0` when the normal
  approximation underflows.
- An enormous rise reads as "300x its minimum" instead of "29891.0%", and a
  one-step span reads "step 320" instead of "steps 320-320".
- HTML report: rounding the axis ends outward drew a gridline and tick label
  above the chart frame (the divergence example's loss chart showed a "5"
  over its legend); ticks now stay inside the plotted range. Off-scale value
  labels at the right edge sit beside their markers instead of on the frame.

## [0.1.0] - 2026-09-24

### Added

- Readers for Hugging Face `trainer_state.json`, generic CSV/JSONL metric
  logs, Weights & Biases history CSV exports, PyTorch Lightning
  `CSVLogger` `metrics.csv`, and TensorBoard event files (optional
  `tensorboard` extra).
- Nine detectors: loss spikes, divergence (NaN/Inf and sustained growth),
  plateaus, overfitting onset, LR schedule anomalies, gradient-norm
  explosions/clipping saturation, throughput regressions, eval-metric
  noise, and a loss-floor/leakage heuristic.
- `trainspotter analyze`: terminal, JSON, and self-contained HTML reports;
  `--fail-on warning|error` for CI gating.
- `trainspotter watch`: tails a growing log file and prints new findings
  as they appear.
- Real example logs under `examples/`, generated by training a small
  numpy MLP on scikit-learn's digits dataset with deliberately induced
  pathologies (`examples/generate_examples.py`).
