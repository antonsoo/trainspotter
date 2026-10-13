"""Controlled framing overhead: OLD_SRC NEW_SRC NEW_DIRECTORY.

Both readers use this Python interpreter and its installed TensorBoard version.
The workload is 20,000 SDK events with four scalar readings in each event.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import tensorboard
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

CHILD = """
import hashlib,json,sys,time
from trainspotter.readers import load_run
load_run(sys.argv[1])
times=[]
for _ in range(3):
    start=time.perf_counter()
    run=load_run(sys.argv[1])
    times.append(time.perf_counter()-start)
points={name:[[p.step,p.value,p.wall_time] for p in series.points]
        for name,series in sorted(run.metrics.items())}
print(json.dumps({'seconds':times,'metrics':len(points),'readings':sum(map(len,points.values())),
                  'complete':run.input_complete(),
                  'readings_sha256':hashlib.sha256(json.dumps(points,sort_keys=True).encode()).hexdigest()}))
"""


def main() -> None:
    before, after, directory = map(lambda p: Path(p).resolve(), sys.argv[1:])
    directory.mkdir(parents=True)
    logs = directory / "events"
    writer = EventFileWriter(str(logs))
    for i in range(20_000):
        writer.add_event(Event(wall_time=100 + i / 10, step=i, summary=Summary(value=[
            Summary.Value(tag="loss", simple_value=5 / (1 + i / 100)),
            Summary.Value(tag="eval_loss", simple_value=5.1 / (1 + i / 100)),
            Summary.Value(tag="learning_rate", simple_value=0.001),
            Summary.Value(tag="grad_norm", simple_value=1.1),
        ])))
    writer.close()
    results = {}
    for label, source in [("before", before), ("after", after)]:
        result = subprocess.run([sys.executable, "-c", CHILD, str(logs)],
                                env={**os.environ, "PYTHONPATH": str(source),
                                     "OPENBLAS_NUM_THREADS": "1"},
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        results[label] = json.loads(result.stdout)
        results[label]["reader_sha256"] = hashlib.sha256(
            (source / "trainspotter/readers/tensorboard.py").read_bytes()).hexdigest()
    assert results["before"]["readings_sha256"] == results["after"]["readings_sha256"]
    print(json.dumps({
        "python": platform.python_version(), "tensorboard": tensorboard.__version__,
        "platform": platform.platform(), "available_cpus": len(os.sched_getaffinity(0)),
        "file_bytes": sum(p.stat().st_size for p in logs.glob("*tfevents*")),
        "method": (
            "One warm-up, then three loads per reader; same interpreter and SDK. "
            "Controlled scalar workload, not a production benchmark or a speed guarantee."
        ),
        **results,
    }, indent=2))


if __name__ == "__main__":
    main()
