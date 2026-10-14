# Native bfloat16 scalar readings stopped being unreadable

At baseline `7eebb62`, all five bfloat16 `train/loss` readings in this native
TensorFlow capture became `None` with five skipped events. The float32 controls
loaded correctly. TensorFlow's own decoder read the original event bytes as
`1`, `0.5`, `0.25`, `0.125` and `0.10009765625`.

The capture uses TensorFlow 2.20.0's
[low-level summary writer](https://www.tensorflow.org/api_docs/python/tf/summary/write)
with scalar-plugin metadata and bfloat16 tensors. These are synthetic constants,
not a trained model or evidence about training quality. Ordinary
`tf.summary.scalar` casts to float32 and is not the affected producer path.

TensorBoard's bundled NumPy stub cannot decode the bfloat16 dtype. The source
reader now widens each recorded 16-bit scalar payload directly to float32,
without a TensorFlow runtime dependency. It handles both `tensor_content`
bytes and a single repeated `half_val` bit pattern. Scalar shape is checked
before decoding, and malformed or missing payloads remain unreadable rather
than becoming an assumed zero. NaNs, infinities and signed zero remain
observations. Other tensor dtypes keep their existing SDK decoder.

| Native capture | Baseline | Corrected |
| --- | --- | --- |
| Five bfloat16 loss readings | Five unreadable values | All five match TensorFlow's numeric oracle |
| Float32 controls | Five readable values | Identical values |
| Recorded `0.1` input | Unreadable bfloat16 value | Retains quantized `0.10009765625`; does not recover the original decimal |
| Analyze/watch warning gates | Exit 2: incomplete input | Exit 0 for this valid finite capture |

The [unmodified 754-byte event file](fixtures/events.out.tfevents.native)
has SHA-256 `702db370bb51cf132a66b774d1e7987a660daebd4ea8383e5b2f2825ba7d1e54`.
The [native capture receipt](results/native-capture.json) retains the actual
dtype identifiers, payload bytes, step numbers and TensorFlow-decoded values.
[Before](results/before.json) disagrees in all six library, JSON/HTML and
CLI/watch workflows; [after](results/after.json) disagrees in none.

## Reproduce

```sh
uv sync --locked --extra dev
uv run --no-sync python studies/bfloat16-scalars/replay.py --src src --out /tmp/bfloat16.json
```

Export `7eebb62` to another directory and pass its `src/` to that replay with
`--observe-only`. The replay checks product import origin, hashes the runtime
reader and original event bytes, invokes the actual CLI in a separate working
directory and retains its full JSON/terminal captures. Single-file CLI calls
use the existing explicit `--format tensorboard` option.

To generate a new native capture and exhaustive numeric oracle independently:

```sh
uv run --no-project --python 3.12 --with tensorflow-cpu==2.20.0 python studies/bfloat16-scalars/capture.py --out /tmp/new-bfloat16-capture --all-words
```

Use a new output directory. Event timestamps and filenames vary; numerical
expectations are reproducible. The permanent producer reproduced all original
numeric expectations and the exhaustive oracle bytes in this session.

The [all-pattern oracle receipt](results/all-words-oracle.json) identifies a
TensorFlow `make_ndarray` decode of all 65,536 bfloat16 bit patterns, widened
to float32 and retained as a small [compressed table](fixtures/native-all-words.f32le.gz).
Both scalar payload encodings match its numeric values, including negative
zero, infinities and NaNs. NaN payload/signalling distinctions are not part of
the Python numeric comparison. This is codec coverage, not 65,536 training runs.

The regressions also check missing/short/long binary data, absent/duplicate or
out-of-range half values and a non-scalar shape. They keep these readings
unavailable and counted as skipped. This does not expand support to arbitrary
tensor shapes or reconstruct pre-quantization values.

The source correction has no corresponding PyPI release or verified hosted
deployment. Package build: `uv build`. Existing sample-site build:
`uv run python scripts/build_site.py`, producing `site/`.
