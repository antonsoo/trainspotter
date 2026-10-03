# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## Unreleased

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
