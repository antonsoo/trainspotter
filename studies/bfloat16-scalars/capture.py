"""Write synthetic low-precision scalars with TensorFlow and retain its decode oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--all-words", action="store_true", help="also retain an exhaustive numeric codec oracle"
    )
    args = parser.parse_args()
    import tensorflow as tf
    from tensorflow.core.framework.summary_pb2 import SummaryMetadata

    args.out.mkdir(parents=True, exist_ok=False)
    metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="scalars"))
    writer = tf.summary.create_file_writer(str(args.out))
    expected = []
    try:
        with writer.as_default():
            for step, value in enumerate([1.0, 0.5, 0.25, 0.125, 0.1]):
                scalar = tf.constant(value, dtype=tf.bfloat16)
                assert bool(
                    tf.summary.write(
                        "train/loss", scalar, step=step, metadata=metadata.SerializeToString()
                    )
                )
                assert bool(
                    tf.summary.write(
                        "control/loss",
                        tf.cast(scalar, tf.float32),
                        step=step,
                        metadata=metadata.SerializeToString(),
                    )
                )
                expected.append({"step": step, "value": float(tf.cast(scalar, tf.float32).numpy())})
    finally:
        writer.close()
    files = list(args.out.glob("*tfevents*"))
    assert len(files) == 1
    records = []
    for event in tf.compat.v1.train.summary_iterator(str(files[0])):
        for value in event.summary.value:
            records.append(
                {
                    "tag": value.tag,
                    "step": event.step,
                    "dtype": value.tensor.dtype,
                    "tensor_content_hex": value.tensor.tensor_content.hex(),
                    "tensorflow_value": float(tf.make_ndarray(value.tensor).item()),
                }
            )
    result = {
        "tensorflow": tf.__version__,
        "population": (
            "Synthetic scalar constants written by native TensorFlow summary API; "
            "no model training."
        ),
        "event_bytes": files[0].stat().st_size,
        "event_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "expected": expected,
        "records": records,
    }
    (args.out / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.all_words:
        import gzip

        import numpy as np
        from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

        words = np.arange(65536, dtype="<u2")
        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_BFLOAT16,
            tensor_content=words.tobytes(),
            tensor_shape=tensor_shape_pb2.TensorShapeProto(
                dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=65536)]
            ),
        )
        raw = tf.make_ndarray(tensor).astype("<f4").tobytes()
        compressed = gzip.compress(raw, mtime=0)
        (args.out / "native-all-words.f32le.gz").write_bytes(compressed)
        all_words = {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "population": (
                "All 65536 bfloat16 bit patterns decoded by native TensorFlow make_ndarray "
                "and widened to float32. Numeric NaNs, infinities and signed zero included; "
                "no model training."
            ),
            "words": len(words),
            "uncompressed_bytes": len(raw),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_bytes": len(compressed),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        }
        (args.out / "all-words-oracle.json").write_text(json.dumps(all_words, indent=2) + "\n")


if __name__ == "__main__":
    main()
