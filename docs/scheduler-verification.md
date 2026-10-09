# Scheduler repair verification

Verified locally on 2026-10-09. Runtime code is commit `b2a664f`; the complete
clean checkout used for packaging and browser replay is `1a06548`. The final
verification notes were added afterwards. No push, tag, package release or
deployment was performed.

The [machine-readable record](scheduler-verification.json) includes source and
package hashes, interpreter/dependency versions, retained scheduler browser
results, capture checks and the large-input measurement. The
[usage guide](learning-rate-rebounds.md) explains the behavior and limitations.

## Captures and independent checks

Eight CPU PyTorch optimization runs produced 1,920 observations. Three sparse
derivatives add 72 rows checked by the replay. Each rate was compared against a
closed-form calculation in a verifier that does not import Trainspotter; the
maximum absolute difference was `6.938893903907228e-17`.

The correct resume matches uninterrupted training, including all recorded losses.
The optimizer-only resume and cosine-horizon overrun are newly reported, including
when only every tenth update is retained. Intended cycles remain informational.
Re-running the capture after implementation produced identical hashes for all
11 CSV files used in the before comparison. Their recorded CRLF endings are
preserved by `.gitattributes`, as for the existing timing captures.

[Before](../examples/schedulers/before.json) and
[after](../examples/schedulers/after.json) contain complete LR findings and
input hashes. The before comparison ran a wheel built from the unchanged
`91bd480` checkout. The independent verifier also passed against the repaired
wheel installed in two isolated environments:

| Environment | Python | NumPy | Result |
| --- | --- | --- | --- |
| Current dependency versions | 3.14.7 | 2.5.3 | All 11 CSV replays passed |
| Minimum supported Python and NumPy | 3.10.21 | 1.24.0 | All 11 CSV replays passed |

Neither installed environment used an editable checkout. The wheel's detector
file hash matches the checked source. The original working checkout separately
passed its suite on Python 3.12.12.

## Clean checks and reports

On a detached checkout after `uv sync --locked --all-extras --dev` and `npm ci`:

| Check | Result |
| --- | --- |
| Ruff, including capture and audit scripts | Passed |
| Mypy | Passed, 27 source files |
| Full pytest suite | 404 passed |
| Wheel built from source distribution | Passed |
| Static report-site builder | Index and nine reports built into `site/` |
| Report workflows | 96 passed, including 16 on the scheduler captures |
| Browsers | Chromium 153.0.8010.12 and Firefox 155.0 |
| Layouts | Desktop and 320-pixel mobile; keyboard scrolling and disclosure controls |
| Axe findings, off-origin requests, CSP violations | Zero in those workflows |
| Whitespace and clean-checkout status | Clean |

Node was 26.7.0 and npm 12.0.1. This is local verification, not an Actions run.
The browser fixtures build reports from the captured CSVs; assertions compare
the visible rebound message, informational severity and confirmation step with
the JSON report. Reports open via `file://` with no server or network dependency.

The [desktop](assets/schedulers/resume-desktop.png) and
[375-pixel phone](assets/schedulers/resume-mobile.png) screenshots were opened and
reviewed. The desktop shows the loss, LR and scheduler counter together. The
phone screenshot scrolls the charts to the recorded rebound; findings and
numeric evidence wrap without page overflow.

The commands used for the main checks, from the checkout root:

```sh
uv sync --locked --all-extras --dev
uv run --locked ruff check src tests examples scripts studies/scheduler-rebounds
uv run --locked mypy src
uv run --locked pytest -q
uv build
uv run --locked python scripts/build_site.py
npm ci
npm run test:browser
uv run --locked python scripts/verify_schedulers.py --out /tmp/scheduler-checks.json
```

For an installed-wheel replay, create a separate environment, install the built
wheel, then pass that environment's `trainspotter` executable to
`scripts/verify_schedulers.py --executable`. The exact tested versions are above;
the [capture README](../examples/schedulers/README.md) covers recapturing the
optimizer observations. PyTorch is needed only for recapture, not analysis.

## Large-input check

A controlled 200,000-row CSV contains a linear warmup to `0.1` at step 20,000,
a decline to `0.001` at 120,000, and a rise ending below the first peak. For
integer step `s` from 0 through 199,999, its rate is:

```text
s < 20000:       0.1 * s / 20000
s <= 120000:    0.1 - 0.099 * (s - 20000) / 100000
otherwise:      0.001 + 0.089 * (s - 120000) / 80000
```

The before wheel reports no finding. The repaired wheel retains all 200,000
observations and reports one rise, steps 120,000-199,999. Both browsers opened
the actual generated report and displayed that finding with no page errors.

One local measurement on 14 logical CPUs under WSL2, Python 3.14.7, using
`/usr/bin/time`, including parsing and report output:

| Output | Elapsed seconds | Maximum RSS, KiB | Report bytes |
| --- | ---: | ---: | ---: |
| JSON | 1.19 | 75,096 | 2,461 |
| HTML | 1.31 | 94,256 | 56,659 |

The input was 4,542,865 bytes. This is a scale check, not a controlled performance
comparison or evidence of diagnostic accuracy. Timings vary with the machine
and other work running on it.

## Public logs and scope

The [public-log audit](../studies/scheduler-rebounds/README.md) retains the
results for 4,177 readable files, including 4,052 with usable canonical LR
series. It reports 174 reversals across 82 logs, without asserting their intent.
The 82 unavailable manifest entries and 125 readable files without usable LR
are counted separately. This audit compares only the LR detectors through the
same reader; it does not rerun every loss detector or score scheduler failures.
