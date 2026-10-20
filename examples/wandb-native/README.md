# Unchanged native offline W&B captures

These files are written by W&B 0.31.0 while a NumPy optimizer performs 400
full-batch updates on synthetic eight-feature regression data. There are 100
logging calls per run: training every five updates and evaluation every twenty.
`producer.json` retains the original byte hashes and exact producer versions.
`reference.json` retains actual optimizer losses and the original logging calls.

From a source checkout with `python -m pip install '.[wandb]'`:

```bash
trainspotter analyze examples/wandb-native/unstable-lr/run.wandb --output html --out native.html
```

| Directory | Authored condition | `--fail-on error` result |
| --- | --- | ---: |
| `stable` | Constant stable optimizer learning rate | 0 |
| `unstable-lr` | Learning rate changes from 0.05 to 4.0 at update 160 | 1: recorded divergence |
| `logged-null` | The computed loss at update 240 is deliberately logged as null | 2: missing evidence |
| `custom-counter` | Updates recorded in `optimizer_step`, with SDK chart definitions | 0; logging-axis timing unavailable |

For `custom-counter`, add `--step-key optimizer_step` to use actual training
coordinates; `--require-timing` then succeeds. Without that assertion the same
gate returns 2, even though the SDK uses that field as its chart axis.

Stable optimization still receives heuristic findings, including a measured
timing warning on the short captured CPU run. Runtime clocks include SDK and
logging overhead. These examples are format/integrity controls, not detector
accuracy calibration or a throughput benchmark.

Each capture contains a 100,017-character metadata note at update 100, written
into native summary/history records spanning several 32 KiB blocks. The text is
ignored as a scalar metric. `fragments.json` retains independent decoder offsets.
`framing/zero-first.wandb` is a separate W&B 0.23.1 DataStore-written protocol
control: a zero-length FIRST fragment at a block boundary, followed by LAST.
It is not a recorded training experiment.

The original binaries are unchanged. Host/metadata collection was explicitly
disabled during capture, and the files were checked for machine paths before
retention. No account or cloud sync was used. SDK timestamps mean newly generated
captures have different byte hashes. Re-run the native producer and both
oracles to verify a new capture, rather than expecting timestamped binaries to
be identical. [Full study and reproduction](../../studies/wandb-native/README.md).
