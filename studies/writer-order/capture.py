"""Create labelled SDK ordering controls; no model training is performed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import tensorboard
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer import event_file_writer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    metadata = {}
    for label, stale_bad, latest_bad in [
        ("latest-nonfinite", False, True),
        ("stale-nonfinite", True, False),
    ]:
        directory = ROOT / "fixtures" / label
        directory.mkdir()
        files = []
        with (
            patch.object(event_file_writer, "_global_uid", event_file_writer.AtomicCounter(0)),
            patch.object(event_file_writer.time, "time", return_value=1_790_000_000.0),
        ):
            for counter in range(12):
                writer = event_file_writer.EventFileWriter(str(directory))
                writer.add_event(Event(step=0, session_log=SessionLog(status=SessionLog.START)))
                for step in range(40):
                    value = 1 - step / 80
                    if step == 39 and (
                        (counter == 9 and stale_bad) or (counter == 11 and latest_bad)
                    ):
                        value = float("nan")
                    writer.add_event(
                        Event(
                            step=step,
                            wall_time=200 - counter + step,
                            summary=Summary(value=[Summary.Value(tag="loss", simple_value=value)]),
                        )
                    )
                writer.close()
                files.append(Path(writer._file_name).name)
        metadata[label] = {
            "created_order": files,
            "files": {
                name: {
                    "bytes": (directory / name).stat().st_size,
                    "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                }
                for name in files
            },
            "latest_nonfinite": latest_bad,
            "stale_nonfinite": stale_bad,
        }
    (ROOT / "fixtures" / "controlled-capture.json").write_text(
        json.dumps(
            {
                "tensorboard": tensorboard.__version__,
                "population": (
                    "Actual EventFileWriter frames with controlled same-second clock, "
                    "reset initial counter, synthetic loss readings and explicit session "
                    "restarts; no training run."
                ),
                "captures": metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
