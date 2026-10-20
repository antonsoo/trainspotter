# Read the local training records before a W&B sync

The original source at `740811d` refuses native `.wandb` files. The current
reader feeds their committed scalar histories into the existing diagnostics,
JSON/HTML reports and standalone vectors. It reads the binary file itself and
uses the optional protobuf runtime with pinned persistent schema definitions.
[Exact reader contract](../../docs/wandb-native.md).

## Captures and independent readers

`produce.py` performs 400 NumPy full-batch MSE optimizer updates per case and
logs through `wandb.init(mode="offline")`. All four unchanged native files are
included in [the examples](../../examples/wandb-native/README.md), together with
the exact original hashes and optimizer/log-call references. No account or
sync is used. Metadata/statistics/git/code collection is disabled; the retained
captures contain no machine paths. The data and controls are synthetic.

| Capture | Histories | Complete logical records | Physical fragments | Authored condition |
| --- | ---: | ---: | ---: | --- |
| Stable optimizer | 100 | 311 | 318 | Learning rate 0.05 |
| Unstable learning rate | 100 | 311 | 318 | Learning rate 4.0 from update 160 |
| Logged null | 100 | 311 | 318 | One computed loss deliberately not logged at update 240 |
| Custom counter | 100 | 311 | 318 | `optimizer_step` and native metric chart definitions |

Every run includes a large metadata note in summary/history records. This
exercises real SDK fragmentation, including records over 100,000 bytes. The
note is not treated as a scalar metric.

Two official readers decode the unchanged captures independently:

- W&B **0.23.1 Python DataStore**, with protobuf 6.33.6. Its old reader opens
  `r+b`, so the oracle uses a byte-identical temporary copy and verifies the
  original capture hash afterward. Its fragment counts and record offsets
  are retained in [fragments.json](../../examples/wandb-native/fragments.json).
- W&B **0.31.0 Go LEET**: `leet inspect --no-observability --json` on the actual
  saved file. Its binary SHA-256 is retained with the verification.

Both agree on all **400 history packets** at binary64 scalar precision, with
exact integer coordinates and runtime clocks. One large numeric token is
printed with an exponent by the Python decoder and as an integer decimal by
Go; both produce the same binary64 scalar. `verify.py` compares every normalized
metric point, coordinate and clock against both oracles, as well as the native
record/fragment counts. The complete typed point sets are hashed, rather than
retaining duplicate large metadata strings in the receipt.
[Before-source refusals](results/before.json) |
[Complete verification](results/verification.json).

## Useful and unavailable outcomes

The unstable optimizer actually overflows its recorded loss. The report locates
train-loss divergence at updates **325-400**, and the error gate returns **1**.
These are observed nonfinite readings in complete decoded input. The logged-null
case instead retains an unreadable point at update **240**, breaks the curve
there, and returns **2** for incomplete evidence. It never borrows the separately
retained optimizer reference to fill the missing logged value.

The custom-counter capture's native chart definitions name `optimizer_step`.
The default analysis still identifies `_step` as a logging axis; wall-clock
seconds per training update are unavailable. Selecting `--step-key optimizer_step`
restores the original update range through 400, and `--require-timing` succeeds.
The reader does not infer training units from a display-axis definition.

The stable optimizer receives an actual throughput warning and informational
heuristics in this capture. Its short wall-clock intervals include SDK/logging
overhead. Those findings remain in the report; this study measures reader
fidelity and controlled faults, not detector accuracy or production throughput.

## Damaged evidence

Three clearly authored controls derive from the original stable file:

| Control | Last retained training-loss update | Gate result |
| --- | ---: | ---: |
| Truncate inside the fragmented update-100 history | 95 | 2 |
| Corrupt that fragment's checksum-covered bytes | 95 | 2 |
| Truncate the final record's payload | 400 | 2 |

All retain exactly the independently decoded complete prefix. The remaining
byte count, original boundary and decoding reason stay inspectable. No suffix
is silently joined after a damaged record, and a readable prefix cannot become
a complete-input CI pass. The original SDK files remain unchanged.

`framing.py` uses the actual older SDK writer to leave exactly seven bytes in a
32 KiB block, then writes a nonempty history. The SDK emits an empty FIRST
fragment and a LAST in the next block. The reader retains its one scalar:
two logical records, three physical fragments. This is an authored protocol
boundary control, separate from the recorded optimizer runs.

## Native types are evidence

Review of the unpublished reader at `ff96e21` found that reusing CSV coercion
turned native JSON strings into measurements. `capture_types.py` records an
actual SDK-written control with loss values `1`, `"0.5"`, `"NaN"`, `0.1`.
The independent older SDK decoder retains the exact native `value_json` tokens
and types. The original reader invented a numeric 0.5 and an observed NaN,
then returned divergence gate 1. The corrected reader keeps two gaps, makes no
divergence finding and returns incomplete-evidence gate 2.
[Raw oracle](results/types-oracle.json) |
[Unpublished before](results/types-before.json) |
[Corrected reader](results/types-after.json).

`verify_types.py oracle FILE OUT` runs in the older SDK environment;
`verify_types.py verify FILE OUT --oracle ORACLE` checks the actual API and CLI.
Add `--before` with `PYTHONPATH` pointing to `ff96e21` to replay the coercion
defect. This defect was found before publication; PyPI does not read this format.

The producer now calls `wandb.teardown()` after `run.finish()` before taking a
file snapshot. A short SDK log can remain buffered until the shared service
closes. All four original optimizer captures were separately byte-matched to
their final SDK files after the producer process exited. Their unchanged
manifest's producer hash identifies the original script preserved at `762fae6`.
The type control uses the explicit service-close procedure.

## Local operation

`offline.py` starts a fresh Linux Python process, prohibits socket/socketpair/
connect/sendto/sendmsg in the kernel with seccomp, and installs Python audit
hooks before importing the application. All four captures decode successfully
with no observed Python network or subprocess attempts. The reader does not
start a W&B run or backend. The `wandb` source extra now installs only protobuf;
the persistent descriptors are bundled with their original license notices.

`extract_schema.py` retains the six unmodified persistent descriptors from the
independently installed W&B 0.31.0 SDK in dependency order. The resulting
30,346-byte file has per-descriptor hashes and exact extraction provenance.
The runtime builds the Record class in a private descriptor pool, keeping it
separate from SDK definitions a caller might load. No W&B code is imported.
Unknown fields anywhere in a record refuse against this pinned schema.
The original SDK and Go oracles remain separate from this decoder.

## Reproduction

The initial SDK-backed `a4a88f2` archive built a wheel through its source distribution. All
33 Python module bytes agree across source, sdist and wheel. The full 855-test
suite passes in the locked clean archive and in isolated wheel installations
on Python 3.10.21 / NumPy 1.24.0 and Python 3.15.0 / NumPy 2.5.4. Both wheel
environments reproduce every independent scalar/coordinate/clock check,
native CLI gates, one-shot watch, input overwrite refusal and the Linux
network-prohibited read. Base-only installations give explicit SDK guidance.
[Package receipt](results/package.json) |
[Installed console exports](results/console-vectors.json).

The initial source publication at `b2bed33` was checked independently through
immutable raw GitHub URLs: all 55 changed files match their git object bytes.
[Publication receipt](results/publication.json). This does not publish a PyPI
release or update the hosted workbench.

The actual console HTML exports in both environments have byte-identical SVGs
to the source reports exercised in offline Chromium and Firefox, at 1440 and
375 pixels: 36 report views and 104 downloaded/reopened vectors, including the
type control. The original 25 source vectors stayed byte-identical after the
type fix. Full HTML timestamps are deliberately not treated as stable bytes.
The real desktop capture appears in the repository README; the phone capture
was also visually inspected.

A fresh installed Python 3.15 CLI process read the 250,519-byte unstable capture
in 0.47 seconds with 71,120 KiB peak RSS under GNU time. This is one warm-cache
small-file observation on the shared 14-vCPU WSL2 / Ryzen 7800X3D host with
47 GiB RAM, not a scale benchmark or a process-memory limit.

The producer uses Python **3.12.12**, NumPy **2.5.4**, W&B **0.31.0** and protobuf
**7.36.2**. Use a separate environment with W&B **0.23.1** and protobuf **6.33.6**
for the older official reader and boundary writer. These are verification
dependencies; the application's `wandb` extra requires only protobuf.

```bash
python studies/wandb-native/produce.py /tmp/native-wandb
/path/to/wandb-023/python studies/wandb-native/oracle.py python-sdk /tmp/native-wandb /tmp/sdk-oracle.json
python studies/wandb-native/oracle.py leet /tmp/native-wandb /tmp/leet-oracle.json
python studies/wandb-native/verify.py /tmp/native-wandb /tmp/sdk-oracle.json /tmp/leet-oracle.json /tmp/native-reports /tmp/verification.json
```

SDK timestamps make generated binaries nondeterministic. Keep the original
manifest and run both oracles over each new capture. The retained example
hashes identify these specific originals, not all future reproductions.
Set `PYTHONPATH` to the unchanged baseline `src` and add `--before` for its
refusals. `offline.py CAPTURE OUT` performs the separate Linux network-prohibited
check; it is not a runtime dependency or a Windows/macOS guarantee.

Primary format sources: [persistent protobuf records](https://github.com/wandb/wandb/blob/v0.31.0/wandb/proto/wandb_internal.proto),
[older DataStore](https://github.com/wandb/wandb/blob/v0.23.1/wandb/sdk/internal/datastore.py),
[current LEET inspector](https://github.com/wandb/wandb/blob/v0.31.0/wandb/cli/leet.py).
