# A change of MLflow metric scope is not training divergence

MLflow 3.17.1 writes model-qualified metrics both to a separate model table and
to an unqualified run aggregate. Trainspotter previously called that entire
aggregate `train/loss`, reported complete input and recommended restarting with
a lower learning rate when logging moved to another model/dataset. The native
source does not establish one training trajectory across those scopes.

The corrected reader preserves every recorded run observation under explicit
aggregate names, retains linked scope evidence, and withholds that canonical
diagnosis. It marks affected selected-run keys incomplete. This adds no
dependency and does not reconstruct model histories or infer train/eval roles.

## Controlled native producer

[capture.py](capture.py) uses the real MLflow **3.17.1** client and SQLite backend
on Python 3.12. Values, steps and metric clocks are authored logging inputs;
there is no training, model-artifact download or production history. Six runs
contain 200 native run observations and 100 separate model records. The
[database](native/scopes.sqlite) is retained byte-for-byte, without normalization.
Its [oracle](native/oracle.json) retains the hash, every authored log call, every
SDK-returned history and the actual SQL scope groups. The experiment's artifact
URI and user tag are deliberately portable authored inputs.

| Control | Run observations | Separate model records | Former behavior | Corrected behavior |
| --- | ---: | ---: | --- | --- |
| Unqualified improving loss | 20 | 0 | Complete; no findings | Complete JSON unchanged; other runs do not contaminate it |
| Improving loss then held-out model loss | 40 | 20 | Complete; false divergence error | Incomplete aggregate; no divergence recommendation |
| Identical qualified/unqualified calls | 20 | 20 | Complete | Incomplete; retain all twenty recorded rows |
| Unqualified `train_loss`, qualified `loss` | 40 | 20 | Qualified `loss` claims the canonical alias | Unqualified training alias keeps its twenty readings; model loss stays separate |
| Two models | 40 | 20 | Complete; false divergence error | Incomplete; retain both model identities and all run values |
| Model without a dataset | 40 | 20 | Complete; false divergence error | Incomplete; no invented dataset |

The coalesced-call control matters: forty successful log calls become twenty
run rows. Each has both an unqualified and a qualified call with identical
key/value/step/time. Subtracting model-row signatures would erase the genuine
unqualified observations. Adding the model table would double-count them.

```text
SDK log calls                    Native SQLite
unqualified loss --+
                  +------------> metrics: one run aggregate
qualified loss ---+                  |
                  +------------> logged_model_metrics: scope evidence
                                     |
                               qualify the aggregate name
                               preserve rows; disclose uncertainty
```

`get_metric_history` itself returns these aggregate rows with null model and
dataset qualifiers. Checking its numbers alone would miss the identity defect.
The separate native table provides the missing warning signal. Context already
discarded by the producer remains unrecoverable.

## Actual before/after commands

The [former installed CLI](before/replay.json), built from source revision
`0a8fa452f4ca182e62cd5c0832dd3ed3cc2af113`, matches all 200 SDK observations
numerically but gives three false divergence errors. Both `analyze --fail-on
error` and `watch --once --fail-on error` exit 1 for those cases; all six runs
are reported complete. The [corrected replay](after/replay.json) still matches
all 200 readings exactly, including millisecond clocks. All five scoped runs
now exit 2 from both gates and retain no divergence findings. The unqualified
control exits 0 and its complete canonical JSON is unchanged.

Full per-input JSON remains beside both receipts. Actual CLI HTML is retained
for the mixed-scope and coalesced controls. These logging controls establish a
reader correction, not a new detector-accuracy estimate.

From this checkout:

```sh
uv sync --locked
uv run trainspotter analyze studies/mlflow-scopes/native/scopes.sqlite --run mixed-dataset --output html --out /tmp/model-scopes.html
uv run trainspotter analyze studies/mlflow-scopes/native/scopes.sqlite --run mixed-dataset --fail-on error
```

The last command intentionally exits 2. Its native step is left as a logging
coordinate: these authored log calls do not measure training updates.
`replay.py --python PYTHON --output DIR --phase after` exercises an actual
selected installation and compares every decoded reading with the SDK oracle.
Recreate the producer in a separate environment with `mlflow==3.17.1`, then
run `capture.py --output NEW_DIRECTORY`; IDs and run metadata times will differ.

## Boundaries

The linked model table is checked in the same read-only transaction as the
selected run history. Other-run records do not change the selected run's scope.
Up to ten model/dataset groups are retained with omitted counts, qualified keys
and available dataset columns. Missing required scope identity columns or
unreadable selected identities refuse the read. Unknown future schemas are not
automatically assumed compatible.

The existing FileStore path already refuses unsupported qualified rows; this
change concerns SQLite run aggregates. A linked model does not by itself prove
that every row came from that model. Scope cannot be recovered by subtraction,
step ordering, dataset names or a model's source-run declaration.

The native behavior is also visible in MLflow's
[log_metric client contract](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.tracking.client.MlflowClient.log_metric).
The retained SDK execution, rather than current documentation alone, establishes
the concrete backend behavior above. See [reader limits](../../docs/mlflow.md).
