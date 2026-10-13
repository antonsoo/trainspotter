# Live watch output through pipes

`trainspotter watch run.jsonl | tee alerts.log` must deliver findings while the
watcher is running. At baseline `1965b988931eed10f5b8da9171a3b612424d4578`, a
five-row synthetic loss log containing NaN produced **no readable pipe bytes for
ten seconds**. Interrupting the process then released both its startup message
and the already-computed error. An ordinary terminal hid this defect by using
line buffering.

The repair explicitly flushes each watch event. It also records a completed
poll's gate status before delivering findings, and a failed read's status before
delivering its retry notice. A consumer that interrupts on those events receives
the corresponding status, rather than the status preceding that event. An early
flush-only attempt exposed this race in a real process: a successfully analyzed
poll returned the initial code 2 when interrupted immediately after its finding.

This is an unreleased source change. No dependency, package version, detector,
report schema, polling interval, selection rule or deduplication rule changed.

## Reproduction

From a checkout with its Python environment installed:

```bash
uv sync --locked --all-extras --dev
python3 studies/live-stdout/replay.py --python .venv/bin/python --source . \
  --work-dir /tmp/trainspotter-live-after --output /tmp/trainspotter-live-after.json
```

Work directories must not already exist. To reproduce the old behavior, extract
`git archive 1965b988931eed10f5b8da9171a3b612424d4578` into a separate directory,
pass that directory as `--source`, and add `--probe-only`. The baseline probe
waits ten seconds before sending SIGINT. Both revisions use ordinary stdout
pipes, with no `-u`, `PYTHONUNBUFFERED`, `stdbuf`, or pseudo-terminal. The replay
removes inherited Python unbuffering and module-path overrides. For an installed
wheel, omit `--source` and pass the fresh environment's Python executable; the
child then runs with `-I`.

The complete replay observes these events while each child is still alive:

```text
missing file -> malformed file -> valid loss + NaN
                                      |
                              append NaN + bad row
                                      |
                              remove bad row
                                      |
                              rewrite an earlier loss -> remove file -> Ctrl-C
```

The appended NaN must not repeat the existing finding. Rewriting earlier data
must print a history notice and evaluate it again. Removing the file after that
successful poll must produce a retry notice and an interrupted exit code of 2.
A separate W&B chart checks
that the automatically chosen run is announced through the pipe. Another process
redirects stdout to an ordinary file, which the parent reads before interruption.
A Unicode filename also exercises the actual UTF-8 stream setup.

Nine completed-command controls cover clean/error `--once`, missing input,
incomplete-input gates, required-but-unavailable timing, JSON reports and plain
terminal output. Their stdout and stderr bytes and exit codes are compared with
the baseline, without normalizing paths or report fields.

## Evidence and limits

`before.json` retains the baseline probe and completed-command controls.
`after.json` retains the repaired source replay. `minimum.json` and `current.json`
repeat the full replay from clean installed wheels. `verification.json` records
the clean build, complete test suite, artifact hashes and equality checks.

The script's elapsed observations include interpreter startup, NumPy imports,
polling and host contention; they are not latency benchmarks or a delivery-time
promise. Explicit flushing removes Trainspotter's own pending text buffer. A
slow or stalled reader still applies pipe backpressure, and downstream programs
may buffer their own output. File visibility is not an fsync durability promise.
The actual process study ran on Linux; it does not establish Windows signal
behavior. Interruptions during an unfinished read or analysis still retain the
most recent completed status. Generated HTML and browser behavior are unchanged
by this CLI-only repair.
