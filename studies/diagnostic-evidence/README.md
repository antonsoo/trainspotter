# Loss-window and gradient-evidence replay

The runtime repair is `5a1c3c2`, compared with baseline `93c74bd`.
[The diagnostic contract](../../docs/diagnostic-evidence.md) explains the
experiments, supported conclusions, and limits.

`before.json` and `after.json` retain complete reports for eleven controlled
inputs, recorded or explicitly derived by
[`capture_diagnostic_evidence.py`](../../examples/capture_diagnostic_evidence.py).
The actual CPU capture used PyTorch 2.8.0+cpu and Python 3.12.12. A second
execution reproduced all eleven CSVs byte-for-byte. The loss trajectories
match their closed forms with maximum absolute differences of
`1.11e-15` (slow) and `2.22e-16` (fast). This is numerical validation of
simple scalar experiments, not a model benchmark.

```bash
uv run python studies/diagnostic-evidence/replay.py --check --output /tmp/after.json
# With the old checkout available:
PYTHONPATH=/path/to/baseline/src uv run python studies/diagnostic-evidence/replay.py --output /tmp/before.json
```

## Cached public histories

[`corpus.json`](corpus.json) compares the two changed detectors using identical
current readers, with files addressed by the revisions in the existing
[`hub-census`](../hub-census/) manifests. The run took 110 seconds locally.
The result records detector hashes, manifest hashes, an aggregate fingerprint
of input IDs and contents, hashes of every changed input, and unavailable
sources. It performs no downloads and does not overwrite the earlier census.

| Slice | Listed | Readable | Unavailable | Readable but incomplete |
| --- | ---: | ---: | ---: | ---: |
| Original census | 3,208 | 3,145 | 63 | 141 |
| Earlier held-out slice | 1,051 | 1,032 | 19 | 44 |
| Total | 4,259 | 4,177 | 82 | 185 |

"Incomplete" uses the current reader's evidence contract, including invalid
values and conflicting steps. These logs are included in the replay with
their limitations; they are not counted as clean. The 82 unavailable files
are listed, not silently dropped into a success count. The historical
"held-out" name identifies the old manifest, not a newly held-out experiment.

| Observed diagnostic change | Before | After |
| --- | ---: | ---: |
| Runs reporting non-finite gradient norms | 109 | 111 |
| Concentration findings with an infinite maximum | 2 | 0 |
| Early-loss findings | 217 | 214 |

The two recovered non-finite gradient findings come from three-reading
histories. Direct inspection confirms NaN at steps 10/20/30 in
`Raiza16/studymate-flant5-lora`, and at step 1 in `Jayasri2006/cricket-qwn`.
The two removed concentration findings have actual infinities in their last
50 observations: `KoSett/large-v2-aug-transcribe` and
`Japhari/sdt-gm4-hesabu-flare-v0.01`. Exact revisions and file paths are in
the result.

Thirteen histories change gradient observations: those four corrections
plus nine changes to explosion groups when NaN/Inf gaps stop connecting
finite segments. Fifteen histories change early-loss decisions: nine
removed, six added, as step positions and valid early windows replace row
fractions. These are changes in evidence and heuristic coverage, **not**
measured improvements in causal accuracy or a new calibration. There are
no human labels for clipping, leakage, or model quality here.

```bash
uv run python studies/diagnostic-evidence/audit.py --output /tmp/corpus.json
```

## Browser and installed artifacts

`gallery.mjs` checks keyboard navigation to the two new reports, their actual
finding counts, accessible rendering, and byte-identical embedded CSV
downloads. It repeats the workflow from local HTML with the browser offline
in Chromium and Firefox, at desktop and mobile widths.

```bash
node studies/diagnostic-evidence/gallery.mjs \
  http://127.0.0.1:8793/ /path/to/site /tmp/gallery.json /tmp/screenshots
```

Verification of the clean checkout, installed wheel, and published site is
recorded in [the verification note](../../docs/verification-diagnostic-evidence-2026-10-09.md).
