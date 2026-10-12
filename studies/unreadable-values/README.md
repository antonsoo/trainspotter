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
reports. [before.json](before.json) records baseline inputs, output, source hashes,
HTML hashes and exit codes. The first case changes exactly the 20 `nan` cells in
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

This is source work. It does not publish a PyPI release, tag or hosted deployment.
