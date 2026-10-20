# Native W&B transaction logs

The source checkout reads one local `.wandb` file through `load_run` or the CLI.
The input is the binary transaction log inside a W&B offline/run directory,
usually named `run-<id>.wandb`. Pass that file rather than the containing directory.
No W&B account, upload, API key or sync is needed to analyze it.

Install from a source checkout with `python -m pip install '.[wandb]'`.
The optional dependency is the protobuf runtime. A 30,346-byte descriptor set
retains the unmodified persistent schema from W&B 0.31.0, with provenance and
upstream notices inside the package. Trainspotter performs read-only file I/O;
it does not install or import the W&B SDK, initialize a run, start a backend
or execute the SDK viewer. The base package still requires only NumPy.
PyPI 0.2.6 predates this source feature.

```bash
trainspotter analyze path/to/run.wandb
trainspotter analyze path/to/run.wandb --output html --out native.html
trainspotter analyze path/to/run.wandb --step-key optimizer_step --output json
trainspotter watch path/to/run.wandb --interval 5
```

Auto-detection uses `.wandb`; `--format wandb-native` selects the same reader
explicitly. `--format wandb` remains the CSV export reader. One native file is
one input run; `--run` and `--chart-single-run` do not reinterpret its identity.
Different nonempty `RunRecord.run_id` values in one stream refuse the read.

## What is read

```text
local file bytes
    |
    v
version-0 header + 32 KiB blocks
    |
    v
CRC-valid FULL or FIRST/MIDDLE/LAST fragments
    |
    v
complete persistent Record.history messages
    |
    +--> recorded scalar values and _runtime clocks
    +--> recorded step coordinates
    +--> gaps at explicitly logged null/nonnumeric values
```

The seven-byte file header is `:W&B`, magic `0xBEE1`, version 0. Record chunks
use IEEE CRC32 over their type byte and payload. This is W&B's format; LevelDB's
masked CRC32C is a different checksum. Padding, fragment ordering, block bounds
and the complete logical record are checked before using any of its values.

The pinned protobuf schema decodes records after those checks. Only committed
scalar histories feed diagnostics. Summaries, system statistics, config and
media/artifact records do not become training measurements. Native run IDs,
header version information, metric definitions, record counts, original history
key paths and a SHA-256 of the complete input file remain in JSON provenance.
The reader follows no sidecar or artifact-file references.

The file is scanned twice so all names are known before canonical metric-name
selection. Each logical record has a 64 MiB limit. Memory also includes the
pinned schema and the scalar series retained for analysis; the block size is not
a process-memory limit. A source changed during reading refuses instead of
mixing two snapshots. Watch retries a changing source on its next poll.

Native `HistoryItem.key` and `nested_key` forms are supported. Nested segments
are joined with dots for display, matching the SDK's viewer. Distinct original
paths that collide at that display name refuse; duplicate items also refuse.
Malformed or duplicate-member value JSON and contradictory native `_step`
representations refuse. No value is chosen by file order to resolve ambiguity.
Unknown protobuf fields, including nested fields, refuse against the pinned
schema. Future producer extensions require a reviewed schema update; they
cannot be silently discarded while declaring the file complete.

## Which horizontal axis?

| Saved evidence | Axis interpretation |
| --- | --- |
| One recognized `train/global_step`, `trainer/global_step`, `global_step` or `step` field | Training-step convention, as in the CSV reader |
| Only `_step` or native `HistoryRecord.step.num` | Logging counter; per-training-step wall-clock timing unavailable |
| `--step-key optimizer_step` (or another exact field) | Caller asserts an integer training-update counter |
| Multiple recognized training counters | Refuse until one is selected |
| Missing/unusable selected coordinate | Skip the history row, retain the incomplete-input count |

When both `_step` and `HistoryRecord.step.num` are present, their numeric values
must agree. A missing `_step` may use the explicitly saved native step message;
no row index is manufactured. All accepted coordinates must be integers within
the existing supported range, `-(2**53 - 1)` through `2**53 - 1`.

`MetricRecord.step_metric` controls the SDK chart's coordinate choice. The raw
definitions are retained, but they do not establish units such as optimizer
updates. Select a custom counter explicitly when it measures training progress.
No metric inherits another row's missing counter inside this reader.

## Missing evidence and damaged files

An absent metric key is ordinary sparse logging. An explicitly logged null is
a known position with no measurement. It stays an unreadable point and breaks
the plotted line and trajectory checks. Nonnumeric readings of a numeric/known
metric also retain their positions. Native NaN/Infinity remain observed values
and can trigger divergence findings.

Native JSON types are preserved. Strings such as `"0.5"` and `"NaN"` cannot
become numeric readings, clocks or coordinates. Finite JSON numbers outside
float64's nonzero finite range become unreadable measurements; conversion must
not invent an infinity or a zero. Observed numeric NaN/Infinity are distinct
from those conversion failures. Text-only metadata remains metadata.

A truncated header/payload/fragment chain or bad checksum stops decoding at the
last complete record. The readable prefix remains available, with
`unread_event_bytes`, the original byte boundary and the decoder's reason.
Later bytes are not joined across the damaged region. The complete file is
still fingerprinted. `--fail-on error` and `--require-timing` return **2** for
incomplete input. A complete readable run with an error finding returns **1**
under `--fail-on error`. These are different outcomes.

A recorded exit code is process metadata. Input completeness describes decoded
evidence, and findings describe the measured series. An absent exit record leaves
training completion unknown. Supplied run identity, counters and provenance are
not independent certification of the original training process.

JSON reports and standalone SVG downloads retain native file identity. An SVG
also retains the unread byte boundary for partial input, along with the existing
axis, input notes and source labels. It is a displayed trajectory with heuristic
bands; retain the original file and report for full evidence.

## Checked inputs

[The native study](../studies/wandb-native/README.md) captures actual NumPy
optimizer updates using W&B 0.31.0 offline mode. Its two independent decoders are
the older official Python DataStore and the current Go LEET viewer. An additional
SDK-written boundary control checks a zero-length FIRST fragment. A fresh Linux
process reads all captures with kernel socket syscalls prohibited and Python
network/process attempts audited. Hosted and registry publication are separate
from these source and installed-package checks.
