# Resumed sessions and native TensorBoard writer counters

A resumed session can contain the most important observation in a log. It must
not be replaced by an older session just because its filename sorts first.

The native TensorBoard writer appends an unpadded process counter to filenames.
Twelve writers created in one second produce counters 0 through 11. Lexical
filename order places 10 and 11 before 2 through 9. With explicit restart markers,
that can leave writer 9's observations instead of writer 11's.

```text
created: 0 -> 1 -> 2 -> ... -> 9 -> 10 -> 11
lexical: 0 -> 1 -> 10 -> 11 -> 2 -> ... -> 9
```

The unmodified `fixtures/native` capture was produced by actual TensorBoard 2.21.0
`EventFileWriter` calls in one real clock second. Each writer records a START at
step zero and one synthetic loss equal to its counter. Before this correction,
the public reader retains loss 9 and reports complete input. The independent SDK
accumulator over the same bytes concatenated in recorded creation order retains
loss 11. No model was trained to produce these controls.

The controlled captures add 40 synthetic loss observations per writer and isolate
the same-second condition with a fixed writer clock and initial counter. If only
the latest writer logs NaN, the old CLI incorrectly passes `--fail-on error`. If
only writer 9 logs NaN, it incorrectly fails even though writer 11 replaced that
session. Both analyze and watch-once now agree with creation-order accumulation.
Raw frames, filenames, capture metadata and complete CLI output are retained.

## Ordering evidence and limits

- Reorder only a unique varying counter family with the same timestamp,
  hostname/PID prefix and directory, after validating every member's first
  TFRecord CRCs, version and native writer metadata. Numeric hostname/suffix
  components must not select competing families.
- The header probe reads at most 64 KiB of payload. This bounds metadata
  inspection, not event acceptance: the full reader retains its existing
  physical-length guards and SDK CRC/protobuf validation.
- Keep other families at their existing positions. Do not infer a shared
  chronology from different PIDs, event wall clocks or filesystem times.
- Unknown writer headers, duplicate counters and ambiguous filename families
  retain the existing order. Digit-leading suffixes cannot be distinguished
  from the counter; filenames do not authenticate provenance. Arbitrarily
  renamed/merged exports and process/PID reuse do not establish creation order.
- JSON directory evidence includes `counter_ordered_files` only when the order
  changes. Ordinary report shape remains unchanged. Restart scope and scalar
  decoding remain the existing documented contract.
- TensorBoard's own directory accumulator also uses lexical file order. The
  independent oracle here uses its unchanged accumulator on a single replay
  file in *recorded creation order*, bypassing that directory-order defect.
  Damaged-file controls independently use SDK-validated records from each file
  in the existing fallback order; unread bytes still make configured gates exit 2.

The filename/counter and writer metadata are present in the upstream
[TensorBoard 2.21 writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/summary/writer/event_file_writer.py)
and the supported
[TensorBoard 2.12 writer](https://github.com/tensorflow/tensorboard/blob/2.12.0/tensorboard/summary/writer/event_file_writer.py).

## Reproduce

```sh
uv sync --locked --all-extras --dev
uv run python studies/writer-order/replay.py --src src --out REPLAY_DIRECTORY
```

Use `--src BASELINE/src --observe-only` to observe an older checkout. Omit `--src`
only when running under a separate installed-wheel environment; the driver checks
that the imported module comes from site-packages. Each of 25 cases exercises the
library, actual CLI JSON, terminal gate, watch-once gate and CLI-written HTML.
An HTML command writes its artifact via `--out`; its captured stdout is the real
confirmation, not HTML substituted for terminal output.

`capture.py` regenerates the *controlled* fixtures into fresh directories only;
it does not reproduce the uncontrolled native capture's timestamp or process ID.
The controls cover ascending single-file/under-ten histories, conventional and
numeric suffix/hostname shapes, different timestamp families, manual names,
corrupt/partial headers and split writer directories. No live trainer, TensorFlow
runtime or GPU is needed.

```sh
node studies/writer-order/browser.mjs --before BEFORE_REPLAY --after AFTER_REPLAY --out SCREENSHOT_DIRECTORY
```

The browser replay reads actual CLI-written HTML files offline in Chromium and
Firefox at 1440 and 375 px. It checks both false-pass and false-failure scenarios,
40 retained readings, exact finding presence, accessibility, no page/external
request errors and no document overflow. Eight screenshots show the latest-NaN
case; each was inspected. The mobile report uses its existing chart/table scroll
regions.

This is a source correction. Verification of a local wheel does not constitute a
PyPI release, a version tag or a hosted demo update.
