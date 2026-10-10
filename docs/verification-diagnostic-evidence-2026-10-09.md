# Diagnostic-evidence verification

Verified on 9 October 2026, Pacific time (10 October UTC).
Runtime: `5a1c3c2`. Clean checkout and gallery build: `d3221a9`.
Later documentation and installed-verifier changes do not alter runtime
modules. The package version remains 0.2.6; no PyPI release or tag was made.

## Recorded counterexamples and field replay

- Eleven captured/derived inputs reproduce the false early-loss claim,
  invalid infinite baselines, unsupported clipping inference, and missed
  short-history non-finite norms. Actual PyTorch optimization is separate
  from explicitly injected numerical boundary fixtures.
- A fresh capture reproduces all eleven CSVs byte-for-byte. Dense and sparse
  histories share the exact recorded values. Slow/fast quadratic losses
  match independent closed forms to maximum absolute differences of
  `1.11e-15` / `2.22e-16`.
- The cached public-log replay reads 4,177 of 4,259 listed histories;
  82 remain unavailable and 185 readable inputs are incomplete. It recovers
  non-finite gradient evidence in two short logs and removes two actual
  infinity-based clipping claims. Nine other gradient-group histories and
  fifteen early-loss decisions change under the documented rules.
- The [retained audit](../studies/diagnostic-evidence/corpus.json) includes
  hashes and every changed source. These comparisons establish concrete
  corrections and compatibility boundaries, not causal diagnostic accuracy.

## Clean checkout and wheel

The detached `d3221a9` checkout started without a Python environment or Node
dependencies. `uv sync --locked --python 3.14 --all-extras --dev` and `npm ci`
both completed. Ruff and strict mypy pass; all **512 tests pass** on Python
3.14.7 (18.11 seconds). The working-source run also passed all 512 on Python
3.12.12. `uv build` built the source distribution and then its wheel.

| Installed artifact | Result |
| --- | --- |
| Python 3.10.21, NumPy 1.24.0 | All eleven inputs: JSON, warning gate, and HTML; one-shot watch retains the infinite norm |
| Python 3.14.7, NumPy 2.5.3 | Same results; decoded JSON reports exactly match retained expectations |
| Both environments | 34 actual CLI invocations each, outside the checkout; only NumPy and the local wheel installed |

The [installed verifier](../studies/diagnostic-evidence/installed.py) launches
the environment's installed console entry point, removes `PYTHONPATH`, and
uses an external working directory. It rejects nonstandard JSON numbers,
checks exact reports and exit codes, and writes actual self-contained HTML
files. Results: [minimum stack](../studies/diagnostic-evidence/installed-minimum.json),
[current stack](../studies/diagnostic-evidence/installed-current.json).

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `trainspotter-0.2.6-py3-none-any.whl` | 107,907 | `e207b59bb931127c32332bd310c32a97b4aaa3be206ca0b6e31f354494e1d4a0` |
| `trainspotter-0.2.6.tar.gz` | 142,953 | `0a0a7b725fba3ccdc73fc32247545804faa88af2dfb677766be07f9cc81dd21b` |

## Gallery

The full clean-checkout browser suite passes **168 workflows across 42
reports**, including the new numerical cases and all existing report
fixtures. Chromium and Firefox exercise desktop and 320px mobile layouts,
keyboard scrolling, evidence downloads, accessibility, and the reports'
content security policy without network requests.
[Full browser results](../studies/diagnostic-evidence/browser-clean.json).

The clean build contains 17 reports. Sixteen new-gallery workflows pass in
Chromium 153.0.8010.12 and Firefox 155.0: two reports, two viewport sizes,
served navigation and offline local HTML. Keyboard activation reaches the
right report, finding counts and wording match the expected evidence,
and every embedded CSV download matches the input bytes. No page errors,
external requests, accessibility violations, or page-width overflow were
observed. [Browser results](../studies/diagnostic-evidence/gallery-local.json).

All four retained desktop/mobile screenshots were opened and visually
inspected. The longer gradient explanation wraps on mobile; charts retain
their existing accessible horizontal scrolling.
