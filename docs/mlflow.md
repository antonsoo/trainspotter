# Native local MLflow histories

Trainspotter can inspect `mlflow.db` or a legacy `mlruns/` directory directly.
You do not need to start a tracking server, export one CSV per metric, or
install MLflow in the environment that runs the diagnosis. SQLite is MLflow's
[current default local backend](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/);
FileStore is its older filesystem backend.

This capability is in the source checkout and the sample-report build. **PyPI
0.2.6 predates it.** Use the [source quickstart](../README.md#quickstart).

## Select a run and write an offline report

From the source checkout, this command reads the retained native SQLite capture:

```bash
trainspotter analyze studies/mlflow-native/fixtures/mlflow-3.17.1/mlflow.db --run slow-training --step-key step --output html --out mlflow-report.html
```

Open `mlflow-report.html` from disk. It identifies the selected run, lists the
other run IDs, retains metric names and diagnostic evidence, and offers SVG
chart downloads. Neither the report nor its charts need a server. The
[public sample](https://antonsoo.github.io/trainspotter/mlflow-slowdown.html)
also contains a download of the exact eight-run SQLite input.

For your own database, replace the path and `--run` value. Selection accepts an
exact run ID, an unambiguous ID prefix, or a unique exact display name. Repeated
display names require an ID. With no `--run`, the active run with the most
recorded observations is selected; ties use run-ID order. Use an explicit ID
when selection matters. Deleted runs require explicit selection. JSON retains
IDs, display names, experiment IDs, status, lifecycle and observation counts
for the complete discovered inventory under `run.mlflow`.

Native `step` is a caller-supplied integer; its default can remain zero. It does
not establish optimizer-update counts. Without `--step-key step`, the axis is
labelled logging steps and wall-clock time cannot establish time per training
update. Use the option only when the producer's counter actually measures
training updates. It declares a unit; it does not repair repeated values.
Different loss values at one step make that metric's trajectory ambiguous,
keep its raw evidence available, and fail a `--fail-on` gate with exit 2.

## FileStore and growing stores

Install its metadata decoder in the source environment, then analyze the root,
one experiment directory, or one run directory:

```bash
python -m pip install '.[mlflow-files]'
trainspotter analyze studies/mlflow-native/fixtures/mlflow-3.17.1/mlruns --run overfitting --output html --out mlflow-files-report.html
```

This extra adds PyYAML; it does not install the MLflow SDK. SQLite needs only
the package's existing numpy dependency and Python's `sqlite3` module.

Watch either store with the existing command:

```bash
trainspotter watch mlflow.db --run YOUR_RUN_ID --interval 2 --fail-on error
```

Selection is pinned after the first successful read. A newly logged, larger
neighboring run cannot take over the watch. Each SQLite pass opens a read-only
transaction and reads the full `metrics` table for that run, including committed
WAL records; it never uses `latest_metrics`. Normal SQLite read locks/shared
memory may be involved. It does not checkpoint, migrate, create, or write the
database or its authoritative WAL. An inaccessible or incompatible snapshot is
unavailable input, not a clean diagnosis.

FileStore has no transaction spanning all metric files. Each pass records file
identity and byte length, then reads that prefix. Appends beyond it wait until
the next poll. A partial final line is reported as incomplete; replacing or
truncating a snapshotted file refuses the read. This bounds ordinary append
races, but does not promise detection of arbitrary in-place edits or a single
atomic time across the whole directory. Native deleted experiments in `.trash`
remain discoverable; their runs are not default candidates.

Reports cannot replace the source database, its SQLite sidecars, or discovered
FileStore metrics, tags, parameters and run/experiment metadata. This includes
aliases and future files under metric/tag/parameter directories. An existing FileStore
`artifacts/` directory can hold a report.

## What is preserved, and what is unavailable

| Evidence | Behavior |
|---|---|
| Native histories | Read all recorded metric observations, rather than the last-value cache; exact duplicate rows already removed by the SQL backend cannot be recovered |
| Timestamps | Native integer milliseconds become seconds, with an exact millisecond round-trip check; unusable clocks remain unavailable while their readable values survive |
| Steps | Exact integers within `[-(2^53-1), 2^53-1]`; unsupported coordinates are rejected rather than rounded |
| NaN | SQL `is_nan` flags restore native NaN observations; malformed values remain unknown and cannot become NaN alerts |
| Infinity | FileStore retains it; MLflow's SQL backend can replace it with a finite extreme, and the original infinity cannot be reconstructed |
| Metric names | Existing normalization keeps one source per canonical metric; nested FileStore metric names retain their complete path |
| Dataset-qualified/model histories | Separate logged-model tables and dataset-qualified FileStore rows are outside this reader's run-history contract; unsupported rows remain incomplete input, never merged into a run curve |
| Schema changes | Unknown metric-table columns or missing required tables refuse the read; no automatic migration |
| Remote stores | HTTP, Databricks, PostgreSQL, MySQL and artifact/model downloads are not supported by this local reader |

The infinity boundary matters in the captured divergence replay: FileStore
retains an infinite loss and triggers the non-finite-loss error gate. The same
SDK-written SQLite run exposes a largest-finite loss, so its report retains
loss-spike warnings and the separately recorded NaN gradient. Converting that
finite number back to infinity would invent evidence. The SQLite report states
this backend boundary. Recorded run status is provenance, not a causal diagnosis.

## Native producer checks

[The retained study](../studies/mlflow-native/README.md) includes stores written
by MLflow **2.17.2** on Python 3.10 and **3.17.1** on Python 3.12. Its oracle is
`MlflowClient.get_metric_history`, captured independently before Trainspotter
reads the files: **34,210 readings across 32 run/backend/version cases**.
Training measurements are replayed from retained MLP and offline W&B captures;
additional default-step, conflicting-step and non-finite cases are controlled.
No new training result, detector accuracy estimate, remote service or model
artifact is claimed. The complete stores and compressed native histories remain
available for replay, with hashes that check source-store preservation.

The [MLflow metric API](https://www.mlflow.org/docs/latest/api_reference/rest-api.html#mlflowmetric)
defines timestamps in milliseconds and steps as integers. Backend behavior is
also checked against the actual installed SDKs, rather than inferred from the
latest documentation alone.
