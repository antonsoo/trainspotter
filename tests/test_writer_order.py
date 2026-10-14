"""Native writer order; the oracle replays the same bytes in recorded creation order."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trainspotter.readers import load_run
from trainspotter.readers.tensorboard import _writer_order

event = pytest.importorskip("tensorboard.compat.proto.event_pb2")
summary = pytest.importorskip("tensorboard.compat.proto.summary_pb2")
writer_module = pytest.importorskip("tensorboard.summary.writer.event_file_writer")
accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")


def capture(directory: Path, *, host: str = "writer.host", suffix: str = "") -> list[Path]:
    directory.mkdir()
    files = []
    # A controlled clock isolates the same-second counter boundary. Files and
    # TFRecords are produced by the actual SDK, without hand-built framing.
    with (
        patch.object(writer_module, "_global_uid", writer_module.AtomicCounter(0)),
        patch.object(writer_module.time, "time", return_value=1_790_000_000.0),
        patch.object(writer_module.socket, "gethostname", return_value=host),
    ):
        for counter in range(12):
            before = set(directory.iterdir())
            writer = writer_module.EventFileWriter(str(directory), filename_suffix=suffix)
            writer.add_event(
                event.Event(step=0, session_log=event.SessionLog(status=event.SessionLog.START))
            )
            writer.add_event(
                event.Event(
                    step=0,
                    wall_time=100 - counter,
                    summary=summary.Summary(
                        value=[summary.Summary.Value(tag="loss", simple_value=float(counter))]
                    ),
                )
            )
            writer.close()
            files.append(next(iter(set(directory.iterdir()) - before)))
    return files


@pytest.mark.parametrize(
    "host,suffix",
    [
        ("writer.host", ""),
        ("writer.1.2.domain", ""),
        ("writer.host", ".native"),
        ("writer.1.2.domain", ".native.123.456"),
        ("writer.host", "_capture"),
    ],
)
def test_native_counter_order_matches_creation_order_oracle(
    tmp_path: Path, host: str, suffix: str
) -> None:
    files = capture(tmp_path / "native", host=host, suffix=suffix)
    replay = tmp_path / "creation-order.tfevents.fixture"
    replay.write_bytes(b"".join(path.read_bytes() for path in files))
    oracle = accumulator.EventAccumulator(str(replay), size_guidance={"scalars": 0}).Reload()
    run = load_run(tmp_path / "native", fmt="tensorboard")
    actual = [(point.step, point.value, point.wall_time) for point in run.get("train/loss").points]
    expected = [(point.step, point.value, point.wall_time) for point in oracle.Scalars("loss")]
    assert actual == expected == [(0, 11.0, 89.0)]
    assert run.input_complete()
    assert run.meta["event_directories"][0]["counter_ordered_files"] == [
        path.name for path in files
    ]


@pytest.mark.parametrize(
    "damage", ["header-crc", "payload-crc", "partial", "huge-length", "writer-name"]
)
def test_unverified_header_does_not_supply_an_order(tmp_path: Path, damage: str) -> None:
    files = capture(tmp_path / "native")
    chosen = files[10]
    raw = bytearray(chosen.read_bytes())
    length = int.from_bytes(raw[:8], "little")
    if damage == "header-crc":
        raw[8] ^= 1
    elif damage == "payload-crc":
        raw[12 + length] ^= 1
    elif damage == "partial":
        raw = raw[:10]
    elif damage == "huge-length":
        raw[:8] = (2**63).to_bytes(8, "little")
    else:
        from io import BytesIO

        from tensorboard.summary.writer.record_writer import RecordWriter

        replacement = event.Event(
            file_version="brain.Event:2",
            source_metadata=event.SourceMetadata(writer="different-writer"),
        )
        buffer = BytesIO()
        RecordWriter(buffer).write(replacement.SerializeToString())
        raw = bytearray(buffer.getvalue() + raw[length + 16 :])
    chosen.write_bytes(raw)
    assert _writer_order(sorted(files)) == sorted(files)


def test_unrelated_families_and_unconventional_names_keep_their_positions(tmp_path: Path) -> None:
    files = capture(tmp_path / "native")
    unrelated = tmp_path / "native" / "events.out.tfevents.1790000000.other.123.5"
    unrelated.write_bytes(files[5].read_bytes())
    renamed = tmp_path / "native" / "events.out.tfevents.manual"
    renamed.write_bytes(files[0].read_bytes())
    original = sorted([*files, unrelated, renamed])
    ordered = _writer_order(original)
    assert ordered.index(unrelated) == original.index(unrelated)
    assert ordered.index(renamed) == original.index(renamed)
    assert [path for path in ordered if path in files] == files
