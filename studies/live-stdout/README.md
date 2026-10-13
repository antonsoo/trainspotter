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
  --work-dir /tmp/trainspotter-live-after --output /tmp/trainspotter-live-after.json \
  --probe-seconds 60
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

The clean archive of implementation `d11b1a1` matched all 445 tracked files.
Locked installation, lint and type checking passed, followed by all 657 tests.
An initial clean process probe hit its ten-second limit during NumPy import,
before the CLI started; that attempt did not test alert delivery. The replay was
repeated with `--probe-seconds 60`. Its actual observation budget is retained in
each receipt. Production code and the completed full-suite checks were unchanged.
The first minimum-Python replay then exceeded a separate thirty-second deadline
on a batch control after completing its live workflows. The study's `--timeout`
now defaults to 120 seconds per stage/command and saves partial results with
`complete: false`; a successful full replay ends with `complete: true`. The
installed replay was repeated with this harness change, using the same wheel.
The baseline and clean-source receipts predate the completion flag; the installed
receipts use it. `verification.json` records both replay-script hashes.

The 118,221-byte wheel rebuilt from the source distribution is byte-identical to
the first wheel. Its SHA-256 is
`5ce7bdfcac72844e6de2bd96b39e141871d1b2bf015f41b704cbe397e92f5fcb`.
Fresh Python 3.10.21 / NumPy 1.24.0 and Python 3.14.7 / NumPy 2.5.4 installs
each match all 29 package files, pass every live workflow and preserve all nine
completed-command outputs and exit codes. Their actual console entry points
also match the isolated module commands. Only `trainspotter/cli.py` differs from
the baseline among the package's runtime files.

The script's elapsed observations include interpreter startup, NumPy imports,
polling and host contention; they are not latency benchmarks or a delivery-time
promise. Explicit flushing removes Trainspotter's own pending text buffer. A
slow or stalled reader still applies pipe backpressure, and downstream programs
may buffer their own output. File visibility is not an fsync durability promise.
The actual process study ran on Linux; it does not establish Windows signal
behavior. Interruptions during an unfinished read or analysis still retain the
most recent completed status. Generated HTML and browser behavior are unchanged
by this CLI-only repair.
