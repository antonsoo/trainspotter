# Chart identity replay

The baseline is `027f8562fc95492481f213d5fe894a2d5fe1141c`. These inputs are
**synthetic**, constructed with Python's CSV writer; they are not hosted W&B
downloads. Each chart has 50 training-loss readings falling from 2 to 1.02 and
50 evaluation-loss readings rising from 0.5 to 1.48. A separate run-history CSV
with explicit `train/loss` and `eval/loss` columns is the diagnostic control.

| Input layout | Previous default | Corrected import |
| --- | --- | --- |
| `runA - train - loss`, `runA - eval - loss` | Two apparent runs; 50 training points retained, evaluation discarded; no overfitting finding | Default exits 2; explicit single-run declaration retains both curves and matches the history control |
| Same metrics, run name containing HTML-significant characters | Same loss of evaluation data | Exact source labels remain literal in the report |
| Run `experiment - resumed`, conventional metric names | Both curves retained by last-separator guess | Default exits 2; exact single-run declaration retains both curves |
| Conventional names, two runs | Selects one run, keeping its curves together | Selection and findings unchanged; report adds original column provenance |

The apparent `runA - eval` selection previously reported the evaluation values
as `train/loss`. It now exits 2. A single-run declaration on the ordinary
two-run fixture also exits 2, naming the foreign column. No partial JSON report
is emitted for these rejected imports.

## Evidence

- [before.json](before.json) and [after.json](after.json): real CLI analyze,
  HTML generation and watch-once results; fixture hashes, complete JSON reports,
  history control and rejected selections.
- [replay.py](replay.py): regenerates the five inputs and compares the complete
  findings and metric summaries to the history control. The four corrected
  chart imports all retain both curves; the previous defaults did so for two.
- [browser.json](browser.json): 16 offline report checks in Chromium 153 and
  Firefox 155, at 1440px and 375px. Both curves, the finding, declaration and exact
  source labels are checked. No HTTP requests, script/CSP errors, page overflow
  or automated WCAG A/AA violations were observed. Automated checks do not prove
  accessibility in every assistive technology.
- [Desktop](../../docs/assets/chart-identity/chromium-1440.png) and
  [phone](../../docs/assets/chart-identity/chromium-375.png) screenshots, plus
  their Firefox counterparts in the same directory. All four were inspected.
  The phone report retains the existing horizontal scrolling for tables/charts.

The reader and CLI regressions also exercise declaration rejection on other
formats, run-name whitespace, nonmetric value columns, empty/colliding metrics,
mutually exclusive options, a growing chart, an added foreign run, recovery and
changed source labels. The full Python suite passes 629 tests.

## Reproduce

```sh
uv sync --locked --all-extras --dev
uv run python studies/chart-identity/replay.py --source src --expect after \
  --artifacts /tmp/trainspotter-chart-after --out /tmp/trainspotter-chart-after.json
npm ci
node studies/chart-identity/browser.mjs /tmp/trainspotter-chart-after \
  /tmp/trainspotter-chart-after.json /tmp/trainspotter-chart-browser.json \
  /tmp/trainspotter-chart-screenshots
```

For the baseline, check out the commit above separately, run the current replay
with `--source /path/to/baseline/src --expect before`, and use a separate output
directory. `--python /path/to/venv/bin/python` without `--source` exercises an
installed wheel. Browser checks use the existing Playwright Chromium/Firefox
installations.

## Limits and compatibility

The CSV cannot prove the caller's single-run declaration. A multi-run chart with
overlapping name prefixes can be byte-identical to a single run with separators
inside metric names. Export one run's history when ownership is uncertain.
Multi-run charts with separator-containing names that previously depended on
the last-separator heuristic now require a different export. This deliberate
compatibility change prevents the reader from manufacturing a run boundary.

W&B recommends conventional metric naming; accepting a supplied label here does
not establish hosted exporter support for arbitrary metric names. Chart sampling,
smoothing, grouped series and the distinction between logging and training steps
are outside this fix. See [the input contract](../../docs/wandb-chart-identity.md).
Source changes remain unreleased; no package release or hosted demo deployment
is part of this study.
