# Historical calibration evidence

This section was retained from the README at source `9d0c62a`. The measured
snapshots and limits below remain historical; no new calibration run is
claimed by the later vector-export feature.

## Calibration study on 3,145 real runs

Many people upload `Trainer`'s checkpoint directories to the Hugging Face Hub,
`trainer_state.json` included, so thousands of real training logs are public.
[**The census**](../studies/hub-census/README.md) collects 3,145 of them, runs
trainspotter over each, and compares its findings with how the run ended.

The original calibration study below predates the later reader, scheduler, and
[evaluation-evidence repairs](evaluation-evidence.md). Its numbers describe
that retained study snapshot, not a fresh accuracy estimate for the current checkout.

The released detectors (0.2.6) did badly at this: they raised a warning or an
error on 75% of the runs that ended at their best loss, and 12% of their errors
were on runs that ended badly. They had been tuned on smooth generated curves,
and a real per-step loss is not smooth. The unreleased detectors in that study were reworked against the corpus, and then scored once on
1,032 further logs from accounts the corpus does not contain:

| | 0.2.6 | Calibration snapshot | Held out: 0.2.6 | Calibration snapshot |
|---|---:|---:|---:|---:|
| Runs that ended at their best loss and were given a warning or an error | 74.6% | 5.7% | 73.0% | 6.9% |
| ... and were given an error | 20.6% | 0.0% | 18.4% | 0.0% |
| Runs that ended badly and were given a warning or an error | 96.1% | 89.9% | 98% | 95% |
| Findings per run | 16.7 | 0.9 | 13.4 | 0.9 |

The first two columns are the 3,145 runs the detectors were tuned on. The zero
is mostly by construction: an error can now come only from the end-of-run
quantities that define a bad ending. What the quieter detectors miss was
measured by damaging 300 held-out runs by known amounts: sustained rises, late
divergence and NaNs are caught 99% of the time or more. A short burst was
missed three times in five at any height, which turned out to be a gap between
two detectors; with it closed, a burst as high as the run's progress or higher
is caught 92% to 98% of the time, and the runs that ended well get the same
warnings as before. [The study](../studies/hub-census/README.md#what-it-no-longer-catches)
has the table.

![The trainspotter report for a public fine-tuning run from the Hub. The training loss falls from 2.7 to about 0.5, spikes to 1.9 at step 3,150, then rises to about 1.4 between steps 14,100 and 27,050 before returning; that stretch is shaded. Two warnings are listed: the spike, at 82% of the run's progress, and the setback, 12,950 steps or 33% of the run.](assets/report-public-run.png)

*A real run from the corpus (the log itself is not redistributed). 0.2.6 gave
it an error and three warnings, each for a single noisy point. This checkout
reports the one spike that undid most of the run's progress for a step, and the
setback: a third of the run spent above a loss level it had already reached.*
