# Source history and evidence

The following notes were moved from the README at source `9d0c62a`. They
describe their own controlled captures. Moving them here does not rerun the
studies or establish a new package release. The [README](../README.md)
introduces the current workflows and installation choices.

**Logging less often must not look like slower training.** In a new controlled
offline W&B capture, the previous checkout reported a **4.95x slowdown** when
the logger changed from every 4 updates to every 20. The source checkout now uses
the recorded `trainer/global_step` and reports **0.991x**. A deliberately delayed
run still reports **2.90x**. When only `_step` is available, the report identifies
logging events and leaves per-training-step timing unavailable. Select a custom
counter with `--step-key`. [Captured records, commands and limits](wandb-axes.md).
These fixes are available from source and are **not yet in the PyPI release**.

The current source also corrects [loss and gradient evidence](diagnostic-evidence.md):
a late low-loss reading does not become "early" because logging is sparse,
constant gradient norms do not establish clipping, and short histories keep
their observed NaN/Inf norms. The retained PyTorch captures include clipped
and unclipped runs with identical pre-clip norm histories.

**NaN/Inf gaps must not invent a loss trajectory.** The current source keeps
spikes, setbacks and terminal-rise comparisons inside finite segments. A
[controlled PyTorch capture](../studies/finite-segments/README.md) previously
produced a 30-step setback and an isolated spike by joining observations across
two logging gaps. Those unsupported claims are gone; the non-finite training
loss still fails the error gate. The alert names the observed loss, since a
NaN logging reduction alone does not prove that optimizer updates diverged.
The source also [bounds possible plateaus by the observed loss level](../studies/plateau-evidence/README.md):
a recovered fourfold rise cannot be folded into one "barely moved" interval.
The report distinguishes the smoothed level from raw fluctuations, retains
noise estimates, and cannot use a future learning rate to hide an earlier stall.
Known-step nonnumeric cells now preserve those boundaries too: dropping
`LOGGER_ERROR` from a loss column could invent a 30-step setback and hide a real
later event. The [unreadable-value repair](../studies/unreadable-values/README.md)
keeps the gaps visible in CLI/JSON/HTML, preserves actual NaN/Inf alerts, and adds
an offline download of unreadable coordinates.
Malformed CSV rows with a readable step now retain those boundaries as well:
[short or extra-width records](../studies/csv-row-gaps/README.md) cannot join separate
loss segments or hide a valid later setback.
Counter-free JSONL now retains rejected records' ordinal positions too. Removing
ten unreadable records between flat loss sections previously produced a
"Diverged" diagnosis and renumbered later observations. The
[JSONL correction and actual reports](../studies/jsonl-record-gaps/README.md) preserve
those boundaries; explicit steps and sparse metric cells keep their existing rules.
These corrections are available from source; no PyPI release was made for them.
