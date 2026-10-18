# Input integrity and report interpretation

Trainspotter's findings concern the observations it could read, under a heuristic
detector set. A report with no findings is not proof that training is healthy.

## Coordinates and names

| Input | Result |
|---|---|
| Exact integer step, including `100.0` or `1e2` | Step 100 |
| Fractional, missing, nonnumeric, boolean, or nonfinite explicit step | Skip and count the row |
| Integer outside `[-9007199254740991, 9007199254740991]` | Skip and count; detectors use double precision |
| No step column anywhere in a CSV/JSONL source | Use row indices and disclose the axis in every report |
| Non-step W&B chart axis | Use disclosed row indices; preserve a time axis as wall time |
| Sparse blank/null metric cell | No observation at that step |
| NaN/Inf metric reading | Retain for diagnostics, including a run with just one loss reading |
| Nonnumeric nonblank cell in a recognized or otherwise numeric metric | Count as invalid; retain its known coordinate as an unreadable boundary |
| CSV row with extra/missing fields and a readable coordinate | Count as a rejected row; retain unknown values for its numeric metrics at that position |
| Rejected JSONL record with no step column anywhere in accepted objects | Count as a rejected record; retain its nonblank record index as an unreadable boundary |
| Text-only metadata column | Ignore as a metric |
| Two names normalize to the same spelling | Keep distinct series; literal names have priority |
| Duplicate CSV header or competing step columns | Reject the source |
| Duplicate JSON keys | Reject the JSON document, or skip that JSONL record |

JSON step numbers are read through decimal parsing before validation so a decimal
close to the representable integer boundary cannot first round into a valid step.
Metric values then convert to floats. An integer metric too large for a float
becomes nonfinite evidence, consistent with an overflowing scientific-notation value.

CSV rows with extra or missing fields are rejected and counted in `skipped_rows`.
When the selected step cell still parses, or the source uses disclosed row indices,
that position remains an unreadable boundary for each recognized or otherwise numeric
metric. None of the malformed row's measurements or timestamps are trusted, even if
they look numeric. These are row rejections, not additional `skipped_values` cells.
An invalid explicit step still locates nothing; no replacement coordinate is invented.
This applies to generic, Lightning, W&B history and W&B chart CSVs.
[Reproduction, native-capture controls and limits](../studies/csv-row-gaps/README.md).

Explicit empty cells in complete-width rows are allowed and remain ordinary sparse cells.
An unterminated quoted field is an input error. Comma, semicolon, and tab delimiters,
UTF-8 BOMs, and decimal commas with non-comma delimiters are supported. All names in
terminal diagnostics are shown with control characters escaped; HTML names are escaped.

Delimiter detection parses complete logical records, including multiline quoted
headers, and samples at most 32 following records for each supported separator.
Single-column guesses cannot override a multicolumn interpretation. A recognized
or explicitly selected coordinate column constrains candidates that have at
least one complete matching record, so a damaged row cannot silently replace a
known axis with punctuation fragments. Equally consistent candidates are refused;
exporting with every field quoted makes the intended separation explicit. An empty
or entirely damaged sample retains the existing header-width fallback and row
diagnostics. This bounded inference cannot prove the intended dialect of every
possible file. [Captured exports and replay](../studies/csv-delimiters/README.md).

Counter-free JSONL uses zero-based positions among **all nonblank records**, including
rejected records. Malformed JSON, repeated members and non-object values (`null`, arrays,
strings or numbers) cannot supply metric measurements. Their known ordinal positions
remain unreadable boundaries, so later readings keep their source positions and loss
trajectories cannot bridge the rejected records. Blank physical lines remain ignored;
valid sparse objects, including `{}` and `{"loss":null}`, keep their ordinary sparse
semantics. A JSON `null` record is not a sparse metric cell inside an object.

If an accepted object supplies a selected step column, the whole file uses that explicit
axis. Rejected records cannot establish a step, even when part of their text resembles
one; no ordinal is substituted. Completing a partial final record replaces its unknown
position on the next read. [CLI, live-watch and offline report evidence](../studies/jsonl-record-gaps/README.md).

## Non-finite loss observations

NaN/Inf loss readings are retained, not discarded as malformed input. A
non-finite training loss remains an error finding and causes `--fail-on error`
to exit 1 in otherwise complete input. Its title is **Non-finite training loss**:
a bad objective, unstable optimizer and an empty logging reduction can produce
the same observed value, so the report does not choose a cause.

Loss spikes and recovered setbacks are assessed separately within contiguous
finite observations. Their smoothing windows, progress, scatter and recovery
endpoints never cross an explicit NaN/Inf reading. Terminal-rise analysis uses
only the final finite segment and runs only if the last observation is finite.
A history ending in NaN cannot be described using an earlier finite endpoint.

For a history containing such gaps, trajectory findings expose
`segment_start_step`, `segment_end_step` and `segment_observations`. Setback
duration fractions use `share_of_segment`, not `share_of_run`; progress is
relative to that segment too. Fewer than ten finite observations cannot
establish a loss level. A later segment can contain a valid event, but is never
called an early-run rise merely because its own first points rose.

Sparse blank/null cells are still absent observations, not explicit boundaries.
This rule cannot infer an unrecorded gap, checkpoint reset, dataset switch or
missing batch from a finite curve. It does not establish real training health
or wasted GPU time. [Controlled capture and before/after evidence](../studies/finite-segments/README.md).

Possible plateaus likewise use finite segments. A nonsignificant slope and
matching endpoints are insufficient: each window and its merged interval must
also fit a bound on the median-smoothed level. The report gives that level's
range separately from raw readings and retains its allowance for observation
noise. This is a heuristic, not a confidence interval for correlated losses.
An LR-based suppression uses only readings at or before the finding's start,
with NaN/Inf/negative LR observations breaking its peak history.
[Examples, sensitivity and limitations](../studies/plateau-evidence/README.md).

## Unreadable values at known coordinates

A nonblank value such as `LOGGER_ERROR` is neither a numeric measurement nor an
ordinary sparse cell. Readers retain its coordinate, and trajectory windows stop
at that position. A level change cannot become a continuous rise across it, and
recovery after it cannot establish a setback duration across the unreadable region.
Independent events inside a later intact segment are still assessed. Direct timing
windows containing unreadable values remain unavailable rather than moving to a
more convenient window.

The Python model represents an unreadable position as `Point.value = None`.
`MetricSeries.values()` maps it to a NaN sentinel for numerical segmentation;
that projection is **not** an observed NaN. Pointwise non-finite diagnostics use
the separately retained actual readings. Equal repeated observed values still
count once. If an unreadable and an observed value share a step, that step breaks
the trajectory without asserting that their unknown values disagree; any actual
NaN/Inf observation at the same step remains reportable. Unequal observed values
continue to exclude the metric's trajectory.

JSON keeps original reading counts in `run.metrics[name].count` and
`analysis_count` (unreadable positions are not counted as numeric readings).
A metric with unreadable positions adds `unreadable_steps`, retaining repetitions.
Its first/last step bounds and `distinct_steps` include those recorded positions. An unreadable value
in repeated-step evidence is JSON `null`; observed NaN remains the string `"nan"`.
The standalone HTML marks unreadable positions with `?`, breaks its plotted line,
and provides a complete offline JSON download of their coordinates. Its table
shows at most 100 metrics and 20 positions per metric; the download is not truncated.

Blank/null sparse cells remain absent observations. These boundaries require a
known metric and usable coordinate. They cannot locate a wholly undecodable row on
an explicit step axis, a missing step, an unrecorded batch or a hidden phase change.
Existing skipped-row and unread-byte counts still make such inputs incomplete;
this repair does not invent positions for them. With no numeric metrics, the CLI still exits 2.

[Reproduced CLI failures, controls and limitations](../studies/unreadable-values/README.md).

## TensorBoard event order

The optional reader uses TensorBoard's TFRecord decoder and protobuf definitions,
without a DataFrame or reservoir sampling. It accepts legacy `simple_value` and
rank-zero numeric tensors with scalar or absent plugin metadata. Text, images,
histograms and non-scalar tensors are outside this reader's scope. A malformed value
explicitly marked as a scalar is counted as skipped, with its known position retained
as an unreadable boundary. Tensor shapes are checked before decoding to avoid
allocating a non-scalar shape. Restart purging removes these boundary positions too;
run selection and overwritten-point counts continue to count only numeric readings.

Files within a writer directory normally use lexicographic filename order, and
records keep their file order. For same-second, same-host/PID TensorBoard
`EventFileWriter` siblings, an unpadded counter can put `.10` before `.9`. A
unique counter family with CRC-valid native writer headers is instead read in
numeric counter order. Other families keep their positions. JSON directory
evidence records `counter_ordered_files` only when the order changes. Wall time
and filesystem times never decide this ordering. Unknown writer headers,
duplicate counters and ambiguous filename families retain the existing order;
this cannot infer creation order for arbitrary renamed or merged exports.
[Native capture, CI gate reproduction and scope](../studies/writer-order/README.md).
For version-2 event streams, every `SessionLog.START` at step T invalidates earlier
observations with step >= T in that same writer directory, including tags which are
never logged again after the restart. For example:

```text
file 1: loss(0)=3 -> loss(10)=99 -> eval_loss(10)=88
file 2: START(5)  -> loss(5)=2

retained: loss(0)=3 -> loss(5)=2
removed:  loss(10), eval_loss(10)
```

A modern step rewind without a START marker retains the observations; no restart is
invented. Duplicate-step readings also remain in the series. Unversioned/version-1
streams follow the older scalar accumulator's per-tag rewind heuristic. Restart scope
does not cross writer directories: a `train/` marker does not purge a `validation/`
file. Two directories claiming the same logical metric get source-qualified names
instead of being concatenated into one curve. That ambiguity makes input incomplete
for a configured gate; selecting one writer directory resolves it.

This contract follows TensorBoard's **legacy scalar EventAccumulator** restart
convention and applies it uniformly to scalar and tensor storage. TensorBoard's
plugin accumulator uses a different convention (its first START is treated as an
initial session), and the legacy accumulator does not purge its tensor reservoir.
Trainspotter does not claim identical output to every TensorBoard dashboard or loader.
Restart tests compare with the unmodified scalar accumulator. For tensor fixtures,
the oracle receives the same event sequence encoded as float32 scalars with exactly
representable values; dedicated tests check tensor decoding and nonfinite readings.

Sources: [scalar accumulator](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py),
[plugin accumulator](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/plugin_event_accumulator.py),
[tbparse's scalar/tensor distinction](https://tbparse.readthedocs.io/en/latest/pages/parsing-scalars.html).
The tests use TensorBoard's real event writer, not a running TensorFlow or PyTorch trainer.
The real committed Transformers log and the generated MLP runs remain regression inputs.

## Repeated metric steps

Readers retain every accepted reading. Detector input is a separate projection:
each metric and step contributes at most one observation. Equal values count once,
including equal observed NaN/Inf values; a nonfinite reading remains diagnostic
evidence. Sparse train/eval rows at the same step belong to different metrics and
do not conflict. Repeated `epoch` bookkeeping values are likewise counted once.

If a metric has unequal values at any step, that entire metric is excluded from
trajectory checks and its connected curve is withheld. The tool does not pick
the last value, average writers or splice the unique portions of different runs.
Other unambiguous metrics remain usable. A repeated equal value with different
or missing wall times has unknown timing; value-only checks can still run.
Value or timestamp conflicts make input incomplete for CI. Equal repeats alone
do not. Modern TensorBoard restart handling happens before this projection.
An excluded loss metric can still contain observed NaN/Inf readings. Those are
reported as **Non-finite readings in ambiguous input**, with distinct-step and
reading counts; the tool does not infer a trajectory from them. CI still returns
2 for incomplete input before applying finding severity.

The original `Run.metrics` and `Run.get()` still return retained readings. Built-in
detectors use `Run.analysis_view()`, including direct `Detector.run()` calls.
JSON adds `run.repeated_steps`, with metric, step, original count, conflict flags
and all decoded value/wall-time/epoch variants and their counts. Metric summaries
retain `count` and add `distinct_steps`, `analysis_count` and `analysis_excluded`.
HTML shows at most 100 groups and 20 variants per group; its standalone download
contains them all. This evidence identifies metric/step readings, not byte offsets
or physical source-line numbers. [Examples and workflow](repeated-steps.md).

## Partial input and CI

TensorBoard's decoder tolerates a truncated trailing record while a writer is active.
Trainspotter accounts for decoded record bytes and framing; remaining bytes are shown
as unread rather than silently treating the report as complete. Corrupt records can
also cause unread bytes. These counts concern the selected run's writer directories;
a damaged sibling run does not invalidate the selection. Re-reading after the writer
completes can recover. Before asking the SDK to decode each record, the reader checks
its advertised length against the bytes available in that local file, including the
16 framing bytes. This prevents a short, damaged tail from asking the standalone
decoder to allocate an enormous buffer. The SDK still validates both checksums;
preceding observations survive, and the unavailable tail remains unread evidence.

Each file read uses its size at the start of that pass. Bytes appended during the
pass remain unread until the next load; `watch` rereads the file. This is not a
memory cap for fully present large records, nor a guarantee against concurrent
in-place rewriting or replacement of the file. The
[controlled replay](../studies/record-lengths/README.md) retains the former
`MemoryError`, partial-write recovery, checksum controls and installed-package checks.

JSON retains `run.event_directories` for every discovered writer, including unrelated
damage: relative directory, observed run names, possible run names, skipped events,
unread bytes, and whether the directory contributes to the selected run's integrity.
Damaged runs with no surviving scalar points remain selectable. Unknown record content
cannot always identify a run: an underscore directory could be an `add_scalars` writer
for its parent. Damage then conservatively applies to either possible owner, including
when no tags survive. Open a known writer directly to narrow the source. The existing
directory-name heuristics still determine run identity; this is not a manifest of
the training process. [Controlled captures and gate replay](../studies/tensorboard-scope/README.md).

The existing JSON schema version 1 has additive fields:

| Field | Meaning |
|---|---|
| `run.input_complete`, `summary.input_complete` | Numeric metrics exist, no rejected supported rows/values, unread event bytes, unresolved source-name collisions or repeated-step value/time conflicts |
| `run.skipped_rows` | Rejected rows, JSONL records, HF entries, or TensorBoard events/scalar values |
| `run.skipped_values` | Invalid nonblank metric cells |
| `run.unread_event_bytes` | Bytes remaining after decoding event records |
| `run.event_directories` | TensorBoard directory evidence and its relation to the selected run |
| `run.overwritten_points` | Scalar observations invalidated by restart semantics |
| `run.restart_markers` | START markers in the selected run's writer directories |
| `run.metric_sources` | Report metric to original name, or writer directory and tag |
| `run.ambiguous_metric_sources` | Logical metric names claimed by multiple writer directories |
| `run.step_column` | Original coordinate column, or a disclosed ordinal axis |
| `run.step_axis` | `step`, `logging_step` or `row_index`; W&B logging positions do not establish training progress |
| `run.step_selection` | `automatic` or `explicit`; `--step-key` asserts a training counter |
| `run.input_notes` | Human-readable limits on the imported observations |
| `run.metrics[name].unreadable_steps` | Optional list of known coordinates whose metric value was unreadable; repeated positions stay repeated |

`input_complete` does not describe training completion, logging coverage, or statistical
power. Deliberately unsupported summaries and sparse metric cells do not make it false.
An explicit restart is a valid input operation, not an incomplete read.

With `--fail-on`, incomplete input returns 2 even if the readable portion contains a
finding which would otherwise return 1. The report is written first. Without a gate,
a readable partial report returns 0. A source with no numeric metrics always returns 2.
Invalid input cannot overwrite an earlier report; report replacement is atomic.

Report destinations must also be distinct from their inputs. For a TensorBoard
directory, this includes every discovered event file, even files belonging to
unselected runs. Direct paths, symbolic links, hard links and aliases through
parent directories are checked before replacement. A new report path inside the
log directory cannot use a filename that this reader would consume as an event
file (`*tfevents.*`, excluding the SDK's `.profile-empty` placeholders). Otherwise
writing a report could corrupt the next directory read even without replacing
an existing event file.

An ordinary report such as `logs/report.html` remains supported and is replaced
atomically. Output-path errors return 2 and leave the source/destination intact.
The protection covers paths managed by `--out` and the default HTML destination;
shell redirection (`> input-file`) opens the destination before Trainspotter runs
and is outside this guard. Filesystem identities are checked at write time; this
does not lock concurrent writers. [Native SDK reproduction and installed-artifact
evidence](../studies/report-inputs/README.md).

## Watch and timing limits

Continuous watch re-reads the source, holds its first selected run by exact name, and
retries missing/malformed reads. If earlier observations change, finding suppression
resets. Growing ranges are not printed on every poll; new findings, reappearances, and
severity changes are. `--once` never waits for a missing file. Poll intervals must be
finite, positive, and at most one day.

Timing comparisons use the original early and recent windows, requiring finite,
positive readings in both. They never discard bad endpoints to move a window into
convenient older data. `1/throughput` is seconds per reported unit; batch-size changes
can change that metric without changing seconds per training step.

The reader retains the selected scalar series in memory and watch reloads the source
each poll; this is designed for thousands to tens of thousands of points, not arbitrary
gigabyte logs. Concurrent writers, renamed event files, absent restart markers and
duplicate steps can leave ambiguous histories. No cross-directory restart is inferred.
