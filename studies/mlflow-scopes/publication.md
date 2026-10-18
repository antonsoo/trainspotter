# MLflow scope correction: source, package and public report

Checked on 2026-10-18. The source fix is published at
`f0061fca61ad5b6f9a9ce869448fb7ca8611cb74`; the sample report is deployed from
Pages revision `f03cf2881792ca78ec464be4d9c6c109979ecd9a`. No PyPI release was made.

| Surface | Evidence |
| --- | --- |
| Clean source clone | Python 3.12.12 locked install, lint/types, 802 unit checks, source distribution and wheel |
| Minimum dependencies | Python 3.10.21, NumPy 1.24.0, TensorBoard 2.12.0, PyYAML 6.0.3, pytest 9.0.3; 802 checks |
| Public Git installation | Fresh pip environment; all 32 Python modules match the clean source and built wheel |
| Native scope controls | Six actual installed CLI replays; all 200 SDK observations retained, all five scoped histories incomplete |
| Existing native controls | All 32 complete canonical JSON reports and gate exits unchanged |
| Existing report workflows | 220 scenarios across Chromium/Firefox, desktop/320 px; keyboard, axe, CSP and network checks |
| Public sample | Twenty generated HTML files plus the canonical root match the clean build exactly |

[Package receipt](package-verification.json), [native control comparison](controls.json),
[existing browser workflows](browser-regression.json), [served bytes](hosted-bytes.json).
The browser regression covers its existing report fixtures. The new aggregate
sample is verified separately below. These are local execution results, not CI claims.

## Actual public workflow

The [public sample](https://antonsoo.github.io/trainspotter/mlflow-model-scopes.html)
is a script-free report computed from the unchanged native SDK database. It
retains the forty run observations, labels their unresolved scope, and gives
no divergence recommendation. Its step remains a logging coordinate.

The [public browser receipt](hosted/browser.json) covers Chromium and Firefox
at 1440 and 375 px. Each workflow starts at the public gallery, opens this
sample, disconnects, uses the keyboard to expose metric sources, and downloads
the exact original SQLite input and an actual standalone SVG. The downloaded
database is parsed by the independent public-Git installation with an enabled
error gate: exit 2, forty retained observations, twenty separate model records
that were not added to the run, and no findings.

Each SVG opens from disk and retains the aggregate name, logging-step axis,
incomplete-input status and source-scope warning in its metadata. Accessibility,
page errors, CSP, unhandled rejections and page overflow checks passed. All four
report screenshots were visually inspected. Phone tables and plots scroll
within their containers.

All four native downloads have the same hash as the retained producer input.
One actual browser copy is kept as [download.sqlite](hosted/download.sqlite);
duplicate identical database copies are omitted. The four actual vector exports
and report screenshots remain beside the receipt.

## Reproduce the public installation

```sh
python -m pip install --force-reinstall "trainspotter @ git+https://github.com/antonsoo/trainspotter.git@f0061fca61ad5b6f9a9ce869448fb7ca8611cb74"
python studies/mlflow-scopes/replay.py --python python --output /tmp/mlflow-scopes-installed --phase after
```

The replay scripts run from this checkout and invoke the selected installation.
The local source still declares version 0.2.6, so `--force-reinstall` replaces an
existing installation of that version. A source installation and precomputed
Pages report are separate from a registry release or an interactive workbench.

The [separate interactive workbench receipt](workbench-publication.json) retains
Hub revision `e4dfb16d477f86767e4e873c58ca78d97fd2ac38`, the sixty-five-file public
audit and four actual browser workflows. All six uploaded native histories
produce complete JSON equal to CPython after disconnection; declaring training
steps does not resolve their model/dataset scope. The receipt also preserves
the original public app's actual false-divergence JSON and its older Hub
revision. These are controlled logging inputs, not new training observations.
