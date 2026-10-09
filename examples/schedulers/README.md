# Captured scheduler behavior

Eight small PyTorch training runs, on the same synthetic binary-classification
data, expose a learning-rate check that used to be unreachable. The CSV files
contain the loss and optimizer learning rate observed immediately before each
update, plus the scheduler's counter. They are actual CPU optimization runs;
the data, missing-state resume and short cosine horizon are deliberate controls.

| File under `captured/` | Experiment |
| --- | --- |
| `linear-control.csv` | 40-update warmup, linear decay through update 240 |
| `linear-resume-restored.csv` | Same schedule; save/reload model, optimizer and scheduler at update 200 |
| `linear-resume-reset.csv` | Same checkpoint, but omit scheduler state when resuming |
| `cosine-control.csv` | Cosine decay with `T_max=240`, 240 updates |
| `cosine-overrun.csv` | Cosine decay with `T_max=120`, still 240 updates |
| `cosine-restarts.csv` | Intended warm restarts every 80 updates |
| `one-cycle-control.csv` | One cycle, with the rise before the decline |
| `step-decay-control.csv` | Intended tenfold drops every 80 updates |
| `*-sparse.csv` | Every tenth observation of the corresponding capture |

`capture.json` records scheduler configuration, resume counters, environment,
capture-script hash and every CSV hash. No trained model or checkpoint is needed
to replay the observations. The correct-resume CSV matches the uninterrupted
one byte for byte, including every recorded loss.

From the repository root, with the source checkout installed:

```sh
uv run --locked trainspotter analyze examples/schedulers/captured/linear-resume-reset.csv
uv run --locked trainspotter analyze examples/schedulers/captured/linear-resume-reset.csv --output html --out /tmp/lr-reset.html
uv run --locked python scripts/verify_schedulers.py --out /tmp/scheduler-checks.json
```

The replay needs only Trainspotter's normal dependencies. Its verifier does not
import Trainspotter: it runs the CLI, checks the captured rates against closed-form
schedule calculations, and checks the reported evidence against the observations.

To recapture the experiments in a separate environment:

```sh
uv venv --python 3.12 /tmp/trainspotter-scheduler-capture
uv pip install --python /tmp/trainspotter-scheduler-capture --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
uv pip install --python /tmp/trainspotter-scheduler-capture numpy==2.2.6
/tmp/trainspotter-scheduler-capture/bin/python examples/schedulers/capture.py --out /tmp/recaptured-schedules
uv run --locked python scripts/verify_schedulers.py --capture /tmp/recaptured-schedules --out /tmp/recaptured-checks.json
```

The retained capture used Python 3.12.12, PyTorch 2.8.0+cpu and one CPU thread.
Floating-point losses can vary with a different environment; the verifier checks
the rates with tolerance and the correct-resume control within that capture.
These experiments establish schedule behavior, not model quality or failure
rates in production. See [the detector and its limits](../../docs/learning-rate-rebounds.md).
