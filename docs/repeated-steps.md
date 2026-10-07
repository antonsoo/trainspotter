# A repeated reading is not another training step

This behavior is available in the source checkout, not the published 0.2.6 package.

One hundred copies of `step=10, loss=1` used to produce a plateau diagnosis at
steps 10–10. There was only one observed step. Trainspotter now keeps all hundred
readings in the evidence and uses one point for analysis.

```sh
uv run trainspotter analyze examples/repeated_steps/single-step.csv --output json
uv run --extra tensorboard trainspotter analyze examples/repeated_steps/replayed
uv run --extra tensorboard trainspotter analyze examples/repeated_steps/conflicting \
  --fail-on warning --output html --out /tmp/conflicting-steps.html
# The last command writes its report and exits 2: input is ambiguous.
```

| Controlled example | Retained loss readings | Analysis points | Input status |
| --- | ---: | ---: | --- |
| One step written 100 times | 100 | 1 | Complete; no plateau |
| 40 TensorBoard events replayed three times | 120 | 40 | Complete |
| 40 events plus a different loss at step 20 | 41 | 0 for loss | Incomplete; loss trajectory excluded |

The TensorBoard files were produced by its actual `EventFileWriter`. They contain
deliberately replayed/conflicting scalar events, not model training. No START
marker resolves the conflicting loss, so the reader retains both 1.8 and 20 at
step 20 (the first value is float32-encoded). Gradient-norm readings remain usable.
The timestamps measure event production, not training speed; any timing finding
on these fixtures describes the artificial producer.

Reproduce the binary captures in a new directory:

```sh
uv run --extra tensorboard python examples/capture_repeated_steps.py /tmp/repeated-capture
```

The script closes each writer and normalizes its one filename afterward; event
bytes are not rewritten, and no multi-file ordering is changed. Tests compare
all retained `(step, value, wall_time)` readings with TensorBoard's own scalar
accumulator. The analysis projection is deliberately stricter than displaying
every accepted event as a connected curve.

In the HTML report, open **Metric sources and point counts** to compare readings
with analysis points. Open **Repeated-step evidence** to see the conflicting
values and counts. **Download all repeated-step evidence** works from the saved
file without JavaScript or a server, and includes groups beyond the bounded table.
Single-point charts draw a marker; excluded metrics explain why there is no curve.

To recover, export one writer/run, or provide explicit restart evidence supported
by the source format. Trainspotter cannot infer which same-step value was intended
from file order. It does not average conflicts or silently select a winner. Equal
values with inconsistent wall times keep value-only analysis but no invented time.

JSON retains its schema version and existing reading `count`. The additive
`analysis_count`, `distinct_steps`, `analysis_excluded` and `run.repeated_steps`
fields expose the difference. Nonfinite numeric evidence is encoded as a string
so the document stays standard JSON. See the [input integrity contract](input-integrity.md)
for complete semantics, restart handling and CI precedence.
