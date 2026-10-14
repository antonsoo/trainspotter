"""Native TensorFlow values remain measurements without a TensorFlow runtime."""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path

import pytest

pytest.importorskip("tensorboard")

from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2  # noqa: E402
from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: E402

from trainspotter.readers.tensorboard import _bfloat16_scalar, read_tensorboard  # noqa: E402

STUDY = Path(__file__).resolve().parents[1] / "studies/bfloat16-scalars"
META = summary_pb2.SummaryMetadata(
    plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="scalars")
)


def write(tmp_path: Path, tensors: list[tensor_pb2.TensorProto]) -> None:
    writer = EventFileWriter(str(tmp_path))
    try:
        for step, tensor in enumerate(tensors):
            writer.add_event(
                event_pb2.Event(
                    step=step,
                    wall_time=step + 1,
                    summary=summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(
                                tag="train/loss", tensor=tensor, metadata=META
                            )
                        ]
                    ),
                )
            )
    finally:
        writer.close()


def test_unmodified_native_tensorflow_capture_matches_its_own_oracle() -> None:
    expected = json.loads((STUDY / "results/native-capture.json").read_text())["expected"]
    run = read_tensorboard(STUDY / "fixtures")
    assert [{"step": p.step, "value": p.value} for p in run.get("train/loss").points] == expected
    assert [p.value for p in run.get("control/loss").points] == [p["value"] for p in expected]
    assert run.meta["skipped_events"] == run.meta["unread_event_bytes"] == 0


def test_repeated_half_values_derived_from_native_bits_match_native_oracle(tmp_path: Path) -> None:
    oracle = json.loads((STUDY / "results/native-capture.json").read_text())
    values = [r for r in oracle["records"] if r["tag"] == "train/loss"]
    write(
        tmp_path,
        [
            tensor_pb2.TensorProto(
                dtype=types_pb2.DT_BFLOAT16,
                half_val=[int.from_bytes(bytes.fromhex(r["tensor_content_hex"]), "little")],
            )
            for r in values
        ],
    )
    run = read_tensorboard(tmp_path)
    assert [p.value for p in run.get("train/loss").points] == [
        r["tensorflow_value"] for r in values
    ]
    assert run.meta["skipped_events"] == 0


@pytest.mark.parametrize("encoding", ["bytes", "half"])
def test_nonfinite_and_signed_zero_observations_are_retained(encoding: str, tmp_path: Path) -> None:
    bits = [0x7F80, 0xFF80, 0x7FC0, 0x8000]
    tensors = [
        tensor_pb2.TensorProto(
            dtype=types_pb2.DT_BFLOAT16,
            **(
                {"tensor_content": word.to_bytes(2, "little")}
                if encoding == "bytes"
                else {"half_val": [word]}
            ),
        )
        for word in bits
    ]
    write(tmp_path, tensors)
    run = read_tensorboard(tmp_path)
    values = [p.value for p in run.get("train/loss").points]
    assert values[:2] == [math.inf, -math.inf]
    assert math.isnan(values[2])
    assert values[3] == 0 and math.copysign(1, values[3]) == -1
    assert run.meta["skipped_events"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tensor_content": b"\x80"},
        {"tensor_content": b"\x80\x3f\x00"},
        {"half_val": []},
        {"half_val": [0x3F80, 0x3F00]},
        {"half_val": [-1]},
        {"half_val": [65536]},
    ],
)
def test_missing_or_malformed_scalar_payloads_remain_unreadable(
    payload: dict, tmp_path: Path
) -> None:
    write(tmp_path, [tensor_pb2.TensorProto(dtype=types_pb2.DT_BFLOAT16, **payload)])
    run = read_tensorboard(tmp_path)
    assert run.get("train/loss").points[0].value is None
    assert run.meta["skipped_events"] == 1


def test_vector_shape_stays_unreadable_before_any_scalar_decode(tmp_path: Path) -> None:
    tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_BFLOAT16, half_val=[0x3F80])
    tensor.tensor_shape.dim.add(size=1)
    write(tmp_path, [tensor])
    run = read_tensorboard(tmp_path)
    assert run.get("train/loss").points[0].value is None
    assert run.meta["skipped_events"] == 1


@pytest.mark.parametrize("encoding", ["bytes", "half"])
def test_all_bit_patterns_match_native_tensorflow_decode_oracle(encoding: str) -> None:
    expected_bytes = gzip.decompress((STUDY / "fixtures/native-all-words.f32le.gz").read_bytes())
    assert len(expected_bytes) == 65536 * 4
    for word in range(65536):
        expected = struct.unpack_from("<f", expected_bytes, word * 4)[0]
        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_BFLOAT16,
            **(
                {"tensor_content": word.to_bytes(2, "little")}
                if encoding == "bytes"
                else {"half_val": [word]}
            ),
        )
        actual = _bfloat16_scalar(tensor)
        if math.isnan(expected):
            assert math.isnan(actual), word
        else:
            assert actual == expected, word
            if actual == 0:
                assert math.copysign(1, actual) == math.copysign(1, expected), word
