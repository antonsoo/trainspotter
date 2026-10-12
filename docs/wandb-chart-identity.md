# W&B chart run and metric names

Chart CSV value columns use `<run display name> - <metric name>`. The label has
no separate field that identifies where the run name ends. A separator inside
either name makes the boundary ambiguous.

For example, these synthetic labels have at least two possible interpretations:

| CSV value column | One run named `runA` | Two runs with separate names |
| --- | --- | --- |
| `runA - train - loss` | Metric `train - loss` | Run `runA - train`, metric `loss` |
| `runA - eval - loss` | Metric `eval - loss` | Run `runA - eval`, metric `loss` |

Previously, Trainspotter always took the last separator. In the first
interpretation, it discarded one of the two curves by default. Selecting the
apparent `runA - eval` run then interpreted its `loss` as training loss. Changing
to the first separator would instead merge genuinely separate runs in the
second interpretation. The CSV bytes alone cannot distinguish them.

## Declare a single-run chart

If the **entire chart was exported from one run**, supply its exact display name:

```sh
trainspotter analyze chart.csv --chart-single-run 'runA'
trainspotter analyze chart.csv --chart-single-run 'runA' --output html --out report.html
trainspotter watch chart.csv --chart-single-run 'runA' --once
```

Every value column must start with `runA - `. Trainspotter removes that exact
prefix and keeps the entire remaining metric name. In the example above,
`train - loss` and `eval - loss` retain their training and evaluation identities.
Names containing separators also work: for a run named `experiment - resumed`,
pass `--chart-single-run 'experiment - resumed'`.

This option asserts the input layout; it does not discover or verify the run's
identity. A multi-run chart whose labels share a prefix can satisfy the syntax
check while violating the assertion. Use it only when you know the whole file
contains one run. Otherwise export one run's history, or export a single-run
chart first. `--run` still selects an existing unambiguous run or unique prefix;
it cannot establish an ambiguous run/metric boundary and cannot be combined with
`--chart-single-run`.

The option is rejected on history CSVs and other input formats. All value
columns are checked, including unrecognized metrics and empty-valued columns.
Band columns remain excluded. Empty metric names and duplicate logical columns
are errors. Watch retains the declaration on every read, skips a changed chart
that violates it, and re-evaluates findings after metric labels change.

## Inspect the interpretation

Terminal and HTML reports disclose the single-run declaration and its limit.
HTML's **Metric sources and point counts** table retains the complete original
chart column. JSON includes `run.chart_identity` with:

- `interpretation`: `declared_single_run` or `unambiguous_labels`;
- `selected_run`: the exact interpreted display name;
- `columns`: metric names mapped to their original CSV column labels.

`run.metric_sources` still maps canonical names such as `eval/loss` to the
original metric name. Chart provenance is additive; other formats do not gain
this field. Ordinary unambiguous multi-run charts retain their prior selection
behavior. Multi-run charts with separator-containing names now require a
different export instead of relying on the previous unverified split.

## Naming and export limits

W&B's [logging documentation](https://github.com/wandb/docs/blob/86eefc342c347f15ce6a4cda3114e9daeaccd403/models/track/log.mdx)
warns that metric names containing spaces or special characters can cause UI and
query problems. Prefer conventional names such as `train/loss` and `eval/loss`.
This reader change makes the interpretation of supplied CSV labels explicit;
it does not establish that every such metric name is supported by W&B's hosted
chart exporter. The reproduction is synthetic, not a downloaded hosted export.

Chart exports can also be sampled or transformed. Resolving a label does not
prove that the curve contains every raw observation. The existing
[W&B axis rules](wandb-axes.md) still apply: `Step` is a logging counter unless
the caller explicitly selects it as a training counter.

This behavior is unreleased and is not in the published 0.2.6 package.
