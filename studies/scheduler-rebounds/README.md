# Learning-rate reversals in the public-log corpus

This audit replays the two pinned manifests from the
[Hub census](../hub-census/README.md) through the old and repaired LR detectors.
Both use the current reader. It measures changed observations, not the accuracy
of calling a schedule faulty: these logs do not establish the intended policy.
No new logs were selected or downloaded for this check.

| Measure | Original corpus | Later corpus |
| --- | ---: | ---: |
| Manifest entries | 3,208 | 1,051 |
| Files read | 3,145 | 1,032 |
| Files unavailable | 63 | 19 |
| Runs with usable canonical `lr` | 3,039 | 1,013 |
| LR observations | 3,999,405 | 1,048,940 |
| Runs with a reported reversal | 60 | 22 |
| Reversal episodes | 119 | 55 |
| All LR findings before | 590 | 188 |
| All LR findings after | 654 | 218 |
| Changed runs with no previous LR finding | 13 | 5 |

There are **174 episodes in 82 logs**, and **94 more LR findings** overall.
Those are different counts: 80 older upward-discontinuity findings are replaced
by a rebound describing the same rise. Existing no-warmup notes and downward
discontinuities remain. Every new finding is `info`; none of these observations
is classified as a proven resume failure or harmful training event.

The later corpus was held out during the earlier loss-detector calibration.
Here it is simply a second existing corpus slice, not a labelled holdout for
schedule correctness. The repair was designed against the unreachable
condition and the controlled optimizer captures before this audit; these
public outcomes were not used to tune the 1% tolerance.

## Examples of newly visible observations

The numeric excerpts below come from cached source files whose hashes and
pinned repository revisions are retained in [results.json](results.json).
They show why staying below the run's maximum does not imply a monotonic decay.

| Source repository | Earlier peak | Low | Subsequent high | Previous LR findings |
| --- | --- | --- | --- | --- |
| `xinyang-hu-00/gpt56-fewshot-seller` | 0.000199703 at step 30 | 0.000000234395 at 380 | 0.0000313586 at 390 | None |
| `Algorithmica/gpt2-foundation-model` | 0.000499998 at step 55 | 0.000428676 at 750 | 0.000496989 at 755 | None |

The table rounds for reading; JSON keeps the decoded numeric values. We did
not obtain their scheduler configurations or observe these training sessions.
Their names are source identifiers, not claims about model behavior or quality.

## Reproduce

From the repository root, after populating the cache as documented in the
Hub census:

```sh
uv run --locked python studies/scheduler-rebounds/audit.py --baseline-ref 91bd480 --out /tmp/lr-public-audit.json
```

The script reads the LR detector at the named Git revision, then runs both
versions on identical current-reader observations. This isolates the change
being assessed; it is not a comparison of two complete installed releases.
`results.json` retains the full baseline commit, both detector hashes, both
manifest hashes, every changed file's content hash, findings and numeric
evidence excerpts. No complete third-party training log is redistributed here.

Unavailable files have explicit source identifiers and error classes. Files
without usable canonical `lr` are counted separately. Neither group counts as
evidence of a clean schedule. Unavailable does not mean that a source repository
currently lacks a log: this audit uses the local cache and makes no network request.

See [learning-rate rebounds](../../docs/learning-rate-rebounds.md) for the
controlled PyTorch captures, thresholds, CLI workflow and limits. The changes
and retained evidence are local and unreleased.
