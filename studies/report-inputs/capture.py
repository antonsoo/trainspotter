"""Capture 20 synthetic loss readings with the installed TensorBoard SDK.

Usage: python studies/report-inputs/capture.py NEW_DIRECTORY
No training job, remote API or external data is involved.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import tensorboard
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=False)
writer = EventFileWriter(str(directory / "writer"))
for step in range(20):
    writer.add_event(
        Event(
            wall_time=1700000000 + step,
            step=step,
            summary=Summary(value=[Summary.Value(tag="loss", simple_value=1 / (step + 1))]),
        )
    )
writer.close()
original = next((directory / "writer").glob("*tfevents.*"))
retained = directory / "events.out.tfevents.controlled"
shutil.copyfile(original, retained)
data = retained.read_bytes()
(directory / "capture.json").write_text(
    json.dumps(
        {
            "tensorboard": tensorboard.__version__,
            "synthetic": True,
            "lossReadings": 20,
            "steps": [0, 19],
            "loss": "float32(1 / (step + 1))",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "note": (
                "Original SDK bytes; filename normalized for the retained copy. "
                "New captures have new writer-header times."
            ),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
