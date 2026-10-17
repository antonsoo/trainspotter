# Native MLflow store replay

A new input capability, not a correction of an earlier MLflow reader.
At source `1195f7b`, the CLI refused the native stores: SQLite input was an
unknown format, and FileStore was treated as a TensorBoard directory with no
event files. The [before receipt](source/before.json) retains all 32 actual
CLI refusals. The [after receipt](source/after.json) checks native history
parity, selected-run reports, integrity gates and declared step timing.

## Population and independent oracle

`capture.py` uses the actual MLflow client to write both native backends. It
then calls `get_metric_history` for each metric and retains the complete
returned observations in `histories.json.gz`, independently of Trainspotter.
`capture.json` records native producer versions, source hashes, per-metric
counts/hashes and original store-file hashes. No third-party training data is
vendored: the training measurements come from this repository's existing MLP
and offline W&B examples.

| Producer | Interpreter | SQLite readings | FileStore readings | Runs per backend |
|---|---|---:|---:|---:|
| MLflow 2.17.2 | Python 3.10.21 | 8,415 | 8,690 | 8 |
| MLflow 3.17.1 | Python 3.12.12 | 8,415 | 8,690 | 8 |

Each backend contains baseline, divergence, overfitting, stable-training,
slow-training, nonfinite, conflicting-steps and default-step runs. MLP clocks
use controlled one-second spacing; W&B wall clocks are rounded to MLflow's
integer milliseconds. The latter preserve the measured delay and changing
logging cadence. Replaying measurements does not train another model or
measure another slowdown.

SQLite removes exact duplicate rows through its native key; FileStore retains
them. Native SQL also clamps infinity to finite extremes. Those differences
are in the native oracle, not repaired or hidden by the reader. NaN flags,
metric names, integer steps and native millisecond clocks match the oracle.

## Replay

From a source environment (add `mlflow-files` for the FileStore half):

```bash
python studies/mlflow-native/replay.py --out /tmp/mlflow-replay
```

The script compares all 34,210 observations, invokes the actual CLI for every
run, checks loss/overfitting/incomplete-input gates, and checks declared training
steps for both cadence controls. Store hashes must remain unchanged. With
`--before-root PATH`, it also records a baseline checkout's refusal. The old
checkout is not modified.

Recreate native stores in a separate SDK environment. Python 3.10 can run the
2.17.2 capture; 3.17.1 was captured with Python 3.12:

```bash
python -m pip install 'mlflow-skinny==3.17.1' sqlalchemy alembic sqlparse
python studies/mlflow-native/capture.py --out /tmp/mlflow-native-new
```

The script disables SDK telemetry and permits its legacy FileStore explicitly.
The controlled artifact URI is provenance only: no model or artifact is written
there. A recreated store has fresh native run IDs; it need not be byte-identical
to the retained capture to test the same native histories.

`live.py --sdk-python PATH --out DIR` starts real CLI watchers in one environment
and writes with the native SDK from another. A new 6,000-reading run must not
steal selection; NaN appended to the selected run must stop the error gate with
exit 1. It then deletes the experiment through the SDK: automatic selection
must refuse it, while explicit deleted-run selection must still load its
recorded history. Only scratch copies are written. See [live receipt](source/live.json)
and its complete watch transcripts.

## Boundaries

See [the reader contract](../../docs/mlflow.md) for SQLite schemas, non-finite
normalization, generic step units, partial FileStore prefixes, dataset-qualified
rows and unsupported remote/model histories. The native histories establish
input fidelity, not general detector accuracy. Source, installed wheel and
hosted report verification are separate checks; no PyPI release is implied.

[Clean installation, wheel replay and offline downloads](verification.md)
retain those separate checks and the actual report screenshots.
