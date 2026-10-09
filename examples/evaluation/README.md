# What an evaluation curve can establish

These are actual CPU PyTorch outputs from controlled synthetic regression,
not curves drawn to satisfy detector thresholds. The model has 16 inputs and
one output, 20 training examples, 128 evaluation examples, and Gaussian label
noise. SGD runs for 300 updates. Evaluation data stay fixed during training.

| Capture | What happened |
|---|---|
| `regression-full.csv` | Training loss falls; evaluation loss reaches its minimum at step 7 and then rises |
| `smooth-six.csv` | First six actual observations; evaluation loss decreases at every update |
| `late-training.csv` | Derived log: evaluation ends at 60 and training readings start at 100; no shared interval |
| `fixed-repeat.csv` | Same frozen model, same evaluation data, 20 identical deterministic passes |
| `fixed-cohort-switch.csv` | Same frozen model, alternating full evaluation set and the 32 highest-loss examples |
| `nan-gap.csv` | Derived log: evaluation at step 100 explicitly replaced by NaN |
| `nan-final.csv` | Derived log: final evaluation explicitly replaced by NaN |
| `predictions.csv` | Raw predictions and targets at eight checkpoints, for independent mean-loss calculation |

For the two frozen-model files, `step` is the **evaluation pass ordinal**;
the model remains at optimizer step 300. They are not continuing-training
histories. The cohort switch is deliberate selection, not random noise.
The missing and invalid readings are explicit manipulations of the captured
log. They do not claim a logger actually dropped or corrupted those values.

`captured/capture.json` records versions, seeds, model fingerprints, selected
example IDs, derivations, and SHA256 hashes. The retained capture used Python
3.12.12 and PyTorch 2.8.0+cpu, one CPU thread and float64. Recreate from the
repository root in a separate environment:

```bash
uv venv /tmp/trainspotter-evaluation --python 3.12
uv pip install --python /tmp/trainspotter-evaluation torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
/tmp/trainspotter-evaluation/bin/python examples/evaluation/capture.py --out /tmp/evaluation-captured
```

The replay requires no PyTorch. It calculates losses with Python arithmetic,
checks the actual predictions against the recorded means, checks the frozen
cohorts independently, and calls the real CLI for every log:

```bash
uv run python scripts/verify_evaluation.py --out /tmp/evaluation-verified.json --reports /tmp/evaluation-reports
```

Use `--capture /tmp/evaluation-captured` to check regenerated captures or
`--executable /path/to/installed/trainspotter` to check a wheel installation.
The retained `before.json` uses the installed CLI from revision `91bd480`;
its two evaluation detector files are byte-identical to `76954b0`, the
pre-repair revision used by the public-log audit. Other detectors are outside
this comparison. `after.json` records the repaired output.

[What changed and its limits](../../docs/evaluation-evidence.md).
