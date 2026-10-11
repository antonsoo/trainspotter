# TensorBoard damage and run selection

An intact run should give the same input-integrity verdict when opened directly
or selected from a parent logdir. At source revision `e0e2c11`, a truncated record
in an unrelated sibling made the selected run incomplete and failed its CI gate.
This checkout scopes integrity counts to the selected run and keeps the other
writers' damage available in `run.event_directories`.

The inputs here are **controlled synthetic scalar histories** written by
TensorBoard's `EventFileWriter`, then deliberately damaged. They are not training
measurements. The replay invokes the actual CLI and reads the library result.

| Case | Before | Corrected |
|---|---|---|
| Direct intact 50-point run | Complete; exit 0 | Complete; exit 0 |
| Same run selected beside damaged sibling | 39 unread bytes; exit 2 | Complete; exit 0 |
| Select damaged sibling | 39 unread bytes; exit 2 | 39 unread bytes; exit 2 |
| Damage selected run's validation writer too | 78 unread bytes; exit 2 | 39 unread bytes; exit 2 |
| Append a checksummed `2**63` record-length header | Uncaught `OverflowError` | 12 unread bytes; 50 points retained; exit 2 |

`before.json` was measured by running this script with the source from `e0e2c11`;
`after.json` was measured with the corrected source. The script creates fresh
files in a temporary directory and removes them after measurement. Filenames are
normalized, but TensorBoard inserts a writer timestamp in its header, so binary
files are not claimed to be byte-reproducible.

```sh
uv sync --locked --all-extras
uv run python studies/tensorboard-scope/verify.py --check
```

The event-file regression checks also exercise empty damaged runs, skipped tensor
scalars, invalid steps, train/validation and `add_scalars` layouts, truncated-tail
recovery, and a one-shot watch gate. The prior TensorBoard accumulator oracle tests
remain responsible for restart and event-order semantics.

Run identity still uses directory conventions. With unread records, an underscore
folder can be its own run or an `add_scalars` writer for its parent; both possible
owners retain the incomplete verdict. An intact selection says only that its
supported recorded input was read, not that the training run finished or is healthy.
These changes are local source work, not a new registry release or deployment.

## Verification of source revision b7fa078

A fresh archive of that commit passed `uv sync --locked --all-extras --dev`,
`ruff check`, `mypy`, all **520 tests**, and `uv build` on Python 3.14.7. A
separate Python 3.12.12 environment installed the built wheel with its TensorBoard
extra and reproduced all five `verify.py --check` cases, including the CLI exits.
The installed environment used TensorBoard 2.21.0; it did not import this checkout.

The clean archive also passed **168 report workflows** using `npm ci` and
`npm run test:browser`: Chromium and Firefox, desktop and 320px mobile, keyboard,
axe, Content-Security-Policy, and no network requests. The report layout is unchanged;
the new evidence is an additive JSON field. No package, site or remote branch was
published by this verification.
