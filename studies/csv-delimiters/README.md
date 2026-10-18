# CSV punctuation must not change a training history

The importer previously chose whichever of comma, semicolon and tab split the
first physical header line into the most fields. A valid comma-separated chart
with an experiment named `seed=0; lr=0.001; wd=0.01` was therefore interpreted as
semicolon-separated. It failed with duplicate column names before reaching the
divergence gate. A quoted multiline header also exceeded the evidence in that
first line.

The correction parses complete logical records with Python's strict CSV reader,
then compares the header width with at most 32 following records per candidate.
It preserves known or declared step columns among candidates with matching rows,
retains damaged-row handling in the actual reader, and rewinds before ingestion.
Explicit `--step-key` also reaches format detection. Equally consistent ambiguous
separators are refused with an instruction to export all fields quoted.

## Retained native exports

[`capture.py`](capture.py) runs actual pandas 3.0.6 `DataFrame.to_csv`, using
the exact literal measurements and sparse cells from the committed
[`examples/divergence.csv`](../../examples/divergence.csv). The source hash,
writer version, per-file hashes, and independent per-metric observation hashes
are in [`capture.json`](capture.json). A separate strict CSV read with the
declared separator verifies every exported field. The writer's quoting rules
are described in [pandas' primary documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).

The labels and layouts are controlled examples, not captured W&B downloads.
Chart MIN/MAX fields repeat center values only to identify the existing chart
layout; they do not measure uncertainty. No new training was run. The three
retained metrics are training loss, evaluation loss and learning rate; other
columns in the original example are outside this study.

| Export | Previous importer | Corrected importer |
|---|---|---|
| Generic control | Divergence gate | Same history and findings |
| Plain chart label | Divergence gate | Same history and findings |
| Chart label containing semicolons | Import error | Same history and findings |
| Chart label containing tabs | Import error | Same history and findings |
| Chart label containing quotes and mixed punctuation | Import error | Same history and findings |
| Chart label containing a line break | Import error | Same history and findings |
| Semicolon writer with comma-rich chart label | Import error | Same history and findings |
| Tab writer with semicolon-rich chart label | Import error | Same history and findings |
| W&B history with punctuation in a metadata name | Divergence gate | Same history and findings |
| Generic CSV with punctuation in a metadata name | Divergence gate | Same history and findings |

The previous source is `5b8b817705d698f700bf8b4e5e61f1eb0ba4ea4f`.
`replay.py` invokes the actual CLI and compares every reading's exact float
representation and integer coordinate with the writer-independent oracle.
Accepted exports must match the generic control's full metric summaries and
findings. Each export retains 323 training-loss readings, 33 evaluation-loss
readings and 323 learning-rate readings. `analyze --fail-on error` and
`watch --once --fail-on error` exit 1 for the observed divergence; the six
previous import failures exit 2.

Focused controls additionally cover all three delimiters across five labels,
explicit multiline coordinate names, an axis word inside metadata, malformed
rows with unknown observations, ambiguous dialect refusal, fully quoted
recovery, and bounded probing with rewind. They preserve incomplete input and
do not let the detector bridge a rejected row.

## Reproduction

```sh
uv run --script studies/csv-delimiters/capture.py
uv run --locked python studies/csv-delimiters/replay.py \
  --source /path/to/previous/src --expect before --out /tmp/csv-before
uv run --locked python studies/csv-delimiters/replay.py \
  --source src --expect after --out /tmp/csv-after
node studies/csv-delimiters/browser.mjs /tmp/csv-after \
  studies/csv-delimiters/capture.json /tmp/csv-browser
```

For an installed wheel, pass `--python /path/to/venv/bin/python` and omit
`--source`. Browser replay opens the CLI's actual standalone HTML reports with
Chromium and Firefox, desktop and 375-pixel widths, offline before navigation.
It verifies findings, original column provenance, keyboard expansion, document
width, accessibility and absence of HTTP requests or script errors.

This study establishes measurement and diagnosis preservation for these
retained exports. Delimiter inference is bounded and cannot establish every
producer's intended dialect. An empty or entirely damaged sample keeps the
existing header-width fallback. This is an import repair, not a new detector
accuracy study or evidence of production training health.

## Clean builds and public workbench

The implementation is `7f6894093461a0a5abeca568ec749df9da4e5263`, pushed
to public Git. A fresh source archive passed Ruff, strict mypy over 32 files,
822 Python checks and a wheel/sdist build on Python 3.12.12 / NumPy 2.5.3.
All 33 runtime files match across source, wheel, sdist and both isolated installs.
The wheel is SHA-256
`906bc3138a2816cbb68bbba3ac4171b1a6cd7c67c5a352dfdc53314075ebe2b6`.

The installed wheel replays all ten exports with no observation mismatch on
Python 3.10.21 / NumPy 1.24.0 and Python 3.14.7 / NumPy 2.5.4. The minimum
environment also passes all 822 Python checks. Forty offline study-report
workflows and all 220 existing browser-report workflows passed. The latter
were selected per engine/layout after interrupted longer invocations; the
tracked browser program was not changed. See [`results/verification.json`](results/verification.json).

The [public workbench](https://huggingface.co/spaces/antonsoloviev/trainspotter/tree/d6ab82a389dcaa78cb560b6efbff1bacfdb3b876)
now pins this engine and uses the same delimiter decision for its bounded
upload preflight. That separate adapter correction is necessary: an engine
re-pin alone still left semicolon and tab exports unreadable there. Fresh
Chromium and Firefox desktop/phone contexts uploaded every retained file and
saved 80 actual JSON/HTML reports, with exact numeric histories and the
documented small tolerance for native/WASM calculated statistics. Ambiguous
input preserves the preceding report. The study and ordinary public audits
record zero page errors and zero HTTP requests during processing after startup.
All 65 published workbench files were verified against the tested bundle,
using immutable Hub card bytes and the host's known data-only entrypoint change.

Receipts distinguish source, local installs, browser reports and public delivery.
This does not publish a new PyPI release or tag; the locally built wheel still
uses the existing package version.
