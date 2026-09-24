"""TensorBoard reader test, against a real event file written with
TensorBoard's own writer (not a hand-rolled protobuf, and not tbparse
itself, so this is an independent round trip: TensorBoard writes, trainspotter
reads via tbparse). Skipped if the `tensorboard` extra isn't installed."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

tb_writer = pytest.importorskip("tensorboard.summary.writer.event_file_writer")
tb_proto = pytest.importorskip("tensorboard.compat.proto.event_pb2")
tb_summary_proto = pytest.importorskip("tensorboard.compat.proto.summary_pb2")


def _write_scalar_events(logdir: Path, tag: str, values: list[float]) -> None:
    writer = tb_writer.EventFileWriter(str(logdir))
    for step, value in enumerate(values):
        summary = tb_summary_proto.Summary(
            value=[tb_summary_proto.Summary.Value(tag=tag, simple_value=value)]
        )
        event = tb_proto.Event(wall_time=time.time(), step=step, summary=summary)
        writer.add_event(event)
    writer.close()


def test_reads_real_tensorboard_event_file(tmp_path: Path) -> None:
    pytest.importorskip("tbparse")
    from trainspotter.readers.tensorboard import read_tensorboard

    logdir = tmp_path / "tb_logs"
    logdir.mkdir()
    values = [2.0 - 0.1 * i for i in range(20)]
    _write_scalar_events(logdir, "loss", values)

    run = read_tensorboard(logdir)
    run.finalize()

    series = run.get("train/loss")
    assert series is not None
    assert len(series) == 20
    assert series.values()[0] == pytest.approx(2.0)
    assert series.values()[-1] == pytest.approx(2.0 - 0.1 * 19)
