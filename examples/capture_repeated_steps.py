"""Produce controlled repeated-step evidence with TensorBoard's real event writer.

uv run --extra tensorboard python examples/capture_repeated_steps.py [OUTPUT_DIRECTORY]
These are deliberately replayed/conflicting scalar events, not model training.
The exact same event is written three times in replayed/. conflicting/ contains
a second value for one previously recorded loss step without a restart marker.
Filenames are normalized after closing; event bytes are not altered.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def capture(destination: Path) -> None:
    if destination.exists() and any(destination.rglob("*tfevents*")):
        raise ValueError("destination already contains captures; choose a fresh directory")
    for name in ("replayed", "conflicting"):
        directory = destination / name
        writer = EventFileWriter(str(directory))
        try:
            for step in range(40):
                event = event_pb2.Event(
                    wall_time=time.time(),
                    step=step,
                    summary=summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(tag="loss", simple_value=2.0 - step / 100),
                            summary_pb2.Summary.Value(tag="grad_norm", simple_value=1.0),
                        ]
                    ),
                )
                for _ in range(3 if name == "replayed" else 1):
                    writer.add_event(event)
            if name == "conflicting":
                writer.add_event(
                    event_pb2.Event(
                        wall_time=time.time(),
                        step=20,
                        summary=summary_pb2.Summary(
                            value=[
                                summary_pb2.Summary.Value(tag="loss", simple_value=20.0),
                            ]
                        ),
                    )
                )
        finally:
            writer.close()
        files = list(directory.glob("*tfevents*"))
        assert len(files) == 1
        files[0].rename(directory / "events.out.tfevents.0000000000.capture")
        print(directory)
    (destination / "single-step.csv").write_text("step,loss\n" + "10,1\n" * 100)


if __name__ == "__main__":
    capture(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/repeated_steps"))
