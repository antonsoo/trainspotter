# Training updates and W&B logging events

Two actual numerical training runs on synthetic data, recorded with W&B 0.22.3
in offline mode. The retained CSVs were decoded from local SDK history records;
they are not downloads from a W&B server. No credentials or remote run are needed.
This reader change is local source work, not a published release.

Each run performs 1,200 logistic-regression updates. The logger commits every
fourth update through 200, then every twentieth update through 1,200. Each update
is deliberately paced to 0.004 seconds; `slow` requests 0.012 seconds after 200.
Actual durations vary. These are controlled demonstrations, not hardware benchmarks.

| Files | Contents |
| --- | --- |
| `*-history.jsonl` | All 101 SDK history protobufs per run, decoded to JSON with the SDK's protobuf definitions |
| `*-training.csv` | Every history key and numeric value, including `_step`, `_runtime` and `trainer/global_step` |
| `*-logging.csv` | The same values, with only `trainer/global_step` removed |
| `*-custom.csv` | The same values, with only that counter renamed `optimizer/updates` |
| `capture.json` | Versions, SHA-256 hashes and all 2,402 independently timed update observations |

`capture/elapsed` is an independent monotonic clock observation logged alongside
each loss. The SDK supplies `_runtime` itself. The verification compares both
clocks' interval medians; CSV values are checked against every retained history
record, and logged steps/losses against the per-update observations.

From a checkout with `uv sync --locked --all-extras --dev` completed:

```sh
# Stable update speed despite changing logger cadence: exit 0.
uv run trainspotter analyze examples/wandb-axes/stable-training.csv --require-timing

# Same observations without a training counter: exit 2, timing unavailable.
uv run trainspotter analyze examples/wandb-axes/stable-logging.csv --require-timing

# Select a custom counter and write a report: exit 1 for the captured slowdown.
uv run trainspotter analyze examples/wandb-axes/slow-custom.csv \
  --step-key optimizer/updates --output html --out /tmp/wandb-slow.html \
  --fail-on warning --require-timing
```

To generate a fresh capture without changing these files:

```sh
uv venv /tmp/trainspotter-wandb-sdk --python 3.12
uv pip install --python /tmp/trainspotter-wandb-sdk wandb==0.22.3 numpy==2.5.3
/tmp/trainspotter-wandb-sdk/bin/python examples/capture_wandb_axes.py \
  --output /tmp/fresh-wandb-axes
uv run python scripts/verify_wandb_axes.py --capture /tmp/fresh-wandb-axes
```

The SDK is pinned because the capture uses its private local-file decoder;
Trainspotter itself has no W&B SDK dependency. The generator's temporary native
transaction files include SDK run metadata and are removed after extracting all
history records. Their hashes are recorded, but the transaction files are not
part of this fixture. No training values or history records are invented by
the CSV conversion. The two column transformations above are explicit controls.

[Before/after results, selection rules and limits](../../docs/wandb-axes.md).
