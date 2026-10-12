# An unreadable value is a boundary, not an absent row

Baseline `95fcb4d749a73ed5fab8b33cb1bd4549221b461d` counted nonnumeric loss
cells as incomplete input, then dropped those positions before analysis. The
remaining values formed a connected curve. This could invent a local spike,
infer a recovery duration through missing evidence, and hide a genuine event
inside a later intact segment.

| Retained input | Baseline trajectory claim | Corrected source |
| --- | --- | --- |
| Text-export fault applied to retained PyTorch capture | Setback 70-100; isolated spikes 70-71 | No cross-gap claim |
| First high value after unreadable region | Loss spike at 70 | No earlier local baseline |
| Recovery after a second unreadable region | Setback 70-100 | No duration through either gap |
| Constant levels 1 and 3 separated by unreadable values | Diverged, steps 0-109 | Separate finite segments |
| Genuine recovered event after a gap | Missed | Setback 100-120, within segment 40-159 |
| Intact finite control | Setback 60-80 | Same finding |
| Original observed-NaN capture | Non-finite training loss, 20 readings | Same alert and error gate |

The incomplete-input gate **already exited 2** for unreadable cells. This repair
changes unsupported diagnostic claims and plotted connections, not a previously
green gate. A real NaN/Inf is still an observation: it retains the non-finite
alert and, for training loss in otherwise complete input, exits 1 under
`--fail-on error`.

## Evidence provenance

[replay.py](replay.py) invokes the actual CLI to create JSON and standalone HTML
reports. [before.json](before.json) and [after.json](after.json) record the baseline and
separately installed wheel: input/output, imported source hashes, HTML hashes
and exit codes. The first case changes exactly the 20 `nan` cells in
`studies/finite-segments/captured/observed.csv` to `LOGGER_ERROR`; every other byte
is unchanged and the pinned original capture hash is checked. This is a
**synthetic text-export fault applied to a retained controlled CPU PyTorch
capture**, not a new training run or a naturally occurring logging incident.
The original capture remains an unchanged positive NaN control. All other
listed inputs are explicitly generated synthetic boundary/control cases.

The [original capture](../finite-segments/README.md) retains optimizer weights
and objective values separately from the faulty log. Its objective was finite
even during the NaN logging reductions. Aggregate logs alone cannot establish
optimizer divergence, exact unobserved recovery time, or wasted GPU time.

## Representation and boundaries

Readers retain unreadable cells at their known coordinates as `Point.value = None`.
The numerical projection uses a NaN gap sentinel; pointwise diagnostics read the
original numeric observations so that sentinel cannot invent a NaN alert. A
repeated unreadable and observed value conservatively breaks the trajectory at
that step while retaining any actual NaN/Inf evidence. Unequal observed values
continue to exclude the metric from trajectory analysis.

Sparse blank/null cells and text-only columns keep their existing meaning.
Unknown columns become metrics only when a numeric reading establishes one;
earlier unreadable positions in those columns are then retained. TensorBoard
scalar decoding failures with known tags/steps retain the same boundary and
follow restart purging. Numeric counts still govern run selection and the
reported number of overwritten readings.

JSON retains numeric reading counts and adds `unreadable_steps` to affected
metrics. First/last bounds and `distinct_steps` include known unreadable positions. HTML draws
separate line segments with `?` markers and offers all unreadable coordinates
as an offline JSON download. Watch distinguishes replacement of an unreadable
value with an actual NaN. Timing does not shift its windows past unreadable
observations. The API's optional point value and numerical projection are
explained in [input integrity](../../docs/input-integrity.md).

This does not place wholly undecodable rows or missing step values on a guessed
axis, detect unrecorded logging gaps, or prove training health. Existing integrity
counters still disclose those separate limitations. No numeric observations is
still an input error. All calculations remain heuristics on the recorded series.

## Inspect the reports

| Case | Before | Installed package |
| --- | --- | --- |
| Text-export fault, desktop Chromium | [Connected curve and unsupported findings](../../docs/assets/unreadable-values/before/chromium-1440-captured-text-fault.png) | [Three segments and exact unreadable positions](../../docs/assets/unreadable-values/after/chromium-1440-captured-text-fault.png) |
| Text-export fault, phone Chromium | [Before](../../docs/assets/unreadable-values/before/chromium-375-captured-text-fault.png) | [After](../../docs/assets/unreadable-values/after/chromium-375-captured-text-fault.png) |
| Valid later setback, desktop Firefox | [Missed event](../../docs/assets/unreadable-values/before/firefox-1440-genuine-after-gap.png) | [20-step setback within the intact segment](../../docs/assets/unreadable-values/after/firefox-1440-genuine-after-gap.png) |
| Valid later setback, phone Firefox | [Before](../../docs/assets/unreadable-values/before/firefox-375-genuine-after-gap.png) | [After](../../docs/assets/unreadable-values/after/firefox-375-genuine-after-gap.png) |

[browser.mjs](browser.mjs) opens the actual generated standalone files with the
browser already offline. The [baseline](browser-before.json) and
[installed-package](browser-after.json) receipts each cover eight workflows:
Chromium/Firefox, 1440/375 px, and the two cases shown above. HTML hashes match
the CLI replay. The corrected reports support keyboard disclosure and a real
JSON download with every recorded unreadable position, retain visible focus,
and have no HTTP requests, script errors, CSP violations, axe violations or
page overflow. All eight retained screenshots were visually reviewed. These
are local-file checks, not hosted-site verification.

## Clean package and compatibility checks

The [artifact receipt](artifacts.json) records clean-archive installation,
package hashes, imported source equality, checked inputs and screenshots.
Python 3.10, 3.12 and 3.14 each pass 578 checks; the declared minimum direct
dependencies on Python 3.10 pass the same 578. Ruff and strict mypy pass. The
full report suite passes 204 Chromium/Firefox desktop/320px workflows, including
keyboard table scrolling, evidence downloads, axe, CSP and network checks.
The separately installed wheel reproduces all seven CLI cases and the eight
specific offline browser workflows above. No environment uses the source
checkout as a substitute for that wheel replay.

## Compatibility replay

The [exact comparison](comparison.json) retains all seven input identities and
before/after findings. The complete NaN and finite-control reports are exactly
equal, and all seven CLI exit codes are unchanged.

The existing cached public-log corpus contains 4,259 revision-pinned inputs:
4,177 yield reports, and 82 remain unreadable. Every complete report/error
record is byte-identical before and after, including finding messages, numerical
evidence, severities and integrity state. The aggregate input hash also matches
the earlier finite-segments study. This reuses historical cached inputs; it is
not a new crawl, calibration set or accuracy estimate.

The corpus and full browser suite use implementation `672f3c0`; the separately
installed wheel uses `89c6b50`. The latter changes only local array annotations
in two Python files. Their syntax trees are equal after erasing local variable
annotations, and all other package source files are byte-identical; the
[comparison receipt](type-only-source-comparison.json) retains both versions.
The annotations fix three baseline mypy errors with Python 3.10's NumPy 2.2
stubs. No numerical operation changes in that follow-up.

## Reproduce

```sh
uv sync --locked --all-extras --dev
uv run --locked python studies/unreadable-values/replay.py --verify --out /tmp/unreadable.json --artifacts /tmp/unreadable-reports
```

The replay checks all seven scenarios. To check a separately installed wheel,
pass `--python /path/to/installed-env/bin/python`; the subprocess records the
actual imported package's source hashes. To reproduce the baseline, extract the
pinned revision with `git archive`, set `PYTHONPATH` to that archive's `src`, and
omit `--verify`. The CLI runs in its artifact directory with the fixture basename as input,
so its actual JSON already has a portable `source_path`; report fields are not rewritten. The HTML hashes cover the actual generated files,
including their wall-clock generation timestamp.

After creating the reports, install browser dependencies and run:

```sh
npm ci
npx playwright install chromium firefox
node studies/unreadable-values/browser.mjs /tmp/unreadable-reports /tmp/browser.json verify /tmp/unreadable-screens /tmp/unreadable.json
```

The corpus uses the existing two `studies/hub-census` manifests and local cache.
Run `scripts/verify_loss_corpus.py --worker` once with `--source` pointing to
each pinned archive's `src`, keeping the same `--cache`, and retain each
`--out` JSONL file. Then compare those records and the seven CLI cases:

```sh
uv run python studies/unreadable-values/compare.py --before studies/unreadable-values/before.json --after studies/unreadable-values/after.json --corpus-before /tmp/corpus-before.jsonl --corpus-after /tmp/corpus-after.jsonl --out /tmp/comparison.json
```

This is source work. It does not publish a PyPI release, tag or hosted deployment.
