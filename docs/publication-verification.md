# Scheduler and evaluation publication verification

2026-10-09. The eight scheduler/evaluation commits ending at `3e5ef24` were
reviewed again before publication. The gallery addition is `e26e11b`; the final
source also directs each timing example to its own capture documentation.
No diagnostic thresholds or formulas changed during this publication pass.
All work was performed without sub-agents.

The gallery adds three inspectable reports: a scheduler reset whose rebound
stays below its old peak, evaluation/training intervals that do not overlap,
and changing evaluation cohorts for a frozen model. Each of the twelve reports
now embeds its exact input log and explains its capture. Downloading an HTML
report retains both the explanation and the input download.

## Independent replay

The [replay ledger](../studies/publication/replays.json) retains hashes and
observed results from this pass.

| Evidence | Fresh result |
| --- | --- |
| Regenerate CPU PyTorch captures with the retained producer scripts | All 19 CSVs byte-identical: eight evaluation files and eleven scheduler files; only capture timestamps changed in metadata |
| Recompute means from individual predictions/targets | 16 means agree within 8.89e-16; 40 frozen-model evaluation passes checked |
| Compare captured learning rates with closed-form schedules | All eleven captures pass; correct resume equals uninterrupted training |
| Replay both public-log audits | Every retained result matches except measured runtime; 4,177 readable cached logs, 82 unavailable entries remain explicit |
| Fresh Python 3.14.7 archive install | Ruff and mypy clean; 419 tests passed in 40.18 seconds |
| Build wheel from source distribution; install outside checkout | Python 3.10.21, NumPy 1.24.0: all seven evaluation and eleven scheduler CLI replays pass |
| Compare installed wheel with clean build | All runtime files identical; hashes retained in the ledger |
| Build gallery from clean archive | Index and twelve reports match the working build after excluding build timestamps |
| Inspect embedded inputs | All twelve downloaded payloads equal the repository's exact source bytes |

The audits compare the same cached inputs and current readers. They are not a
new collection or a causal ground-truth evaluation. The changed diagnosis
counts are retained in the earlier [evaluation](evaluation-evidence.md) and
[scheduler](learning-rate-rebounds.md) studies, including unresolved inputs.

## Browser and artifact checks

Chromium and Firefox passed **124 offline report workflows**, covering desktop
and 320-pixel layouts, keyboard use, accessibility, CSP and no network requests.
The new gallery check adds twelve browser/scenario/viewport combinations over
HTTP and twelve more using `file://`. It compares card counts with report
counts, checks specific diagnostic evidence, and activates input downloads
with the keyboard. Downloaded bytes match the captured CSVs exactly. Neither
format needs remote assets to display the reports or download their input.

The [browser ledger](../studies/publication/browser.json) records the gallery
cases. These [desktop](assets/publication/overlap-desktop.png) and
[phone](assets/publication/cohorts-mobile.png) screenshots were captured from
the generated reports and visually inspected. Capture context wraps on the
phone; charts retain their accessible horizontal scrolling region.

```bash
uv sync --locked --extra dev
uv run --locked python scripts/verify_evaluation.py --out /tmp/evaluation-check.json
uv run --locked python scripts/verify_schedulers.py --out /tmp/scheduler-check.json
uv run --locked python scripts/build_site.py --out-dir site
npm ci
npm run test:browser

# Serve site/ on 4310 first, or supply the hosted gallery URL.
node studies/publication/check-site.mjs
TRAINSPOTTER_SITE_URL=file:///absolute/path/to/site/index.html \
  node studies/publication/check-site.mjs
```

The site build output is `site/`. Source publication and Pages publication are
separate from a package release: the PyPI version remains 0.2.6 and does not
contain these detector repairs. No package, tag, or Hugging Face deployment is
part of this pass. Hosted verification is recorded after the Pages push.
