"""Exercise a real watch process while the native MLflow SDK writes its store.

Copies retained native stores to scratch space. A new, larger neighboring run
must not steal the watch selection; NaN appended to the pinned run must fail
its error gate. The SDK is kept in a separate environment from the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from trainspotter.readers import load_run

ROOT = Path(__file__).resolve().parents[2]
SDK_WRITE = """
import json,sys
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
path, kind, selected, experiment = sys.argv[1:]
uri = ('sqlite:///' + str(Path(path).resolve()) if kind == 'sqlite'
       else Path(path).resolve().as_uri())
client = MlflowClient(tracking_uri=uri)
other = client.create_run(experiment, run_name='larger-neighbor', start_time=1700000000000,
                          tags={'mlflow.user':'trainspotter-native-study'})
for start in range(0, 6000, 500):
    client.log_batch(other.info.run_id, metrics=[Metric('loss', 1/(i+1), 1700000000000+i*1000, i)
                                                for i in range(start, start+500)])
client.set_terminated(other.info.run_id, end_time=1700010000000)
client.log_metric(selected, 'loss', float('nan'), timestamp=1700001000000, step=1000000)
print(json.dumps({'largerNeighbor':other.info.run_id,'appendedTo':selected,'neighborReadings':6000}))
"""
SDK_DELETE = """
import sys
from pathlib import Path
from mlflow.tracking import MlflowClient
path, kind, experiment = sys.argv[1:]
uri = ('sqlite:///' + str(Path(path).resolve()) if kind == 'sqlite'
       else Path(path).resolve().as_uri())
MlflowClient(tracking_uri=uri).delete_experiment(experiment)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    fixture = ROOT / "studies/mlflow-native/fixtures/mlflow-3.17.1"
    manifest = json.loads((fixture / "capture.json").read_text())
    env = {
        **os.environ,
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    with tempfile.TemporaryDirectory(prefix="trainspotter-mlflow-live-") as temp:
        for store in manifest["stores"]:
            selected = max(
                store["runs"],
                key=lambda run: sum(v["observations"] for v in run["histories"].values()),
            )
            path = Path(temp) / store["path"]
            if store["kind"] == "sqlite":
                shutil.copyfile(fixture / store["path"], path)
            else:
                shutil.copytree(fixture / store["path"], path)
            watcher = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "trainspotter.cli",
                    "watch",
                    str(path),
                    "--interval",
                    "0.05",
                    "--fail-on",
                    "error",
                    "--no-color",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            pending = queue.Queue()

            def collect(stream=watcher.stdout, messages=pending):
                for line in stream:
                    messages.put(line)

            reader = threading.Thread(target=collect, daemon=True)
            reader.start()
            lines, deadline = [], time.monotonic() + 30
            try:
                while not any(line.startswith("input    ") for line in lines):
                    assert time.monotonic() < deadline, lines
                    lines.append(pending.get(timeout=max(0.1, deadline - time.monotonic())))
                assert any(
                    line.startswith("run ") and selected["runId"] in line for line in lines
                ), lines
                # Preserve a venv's executable symlink: resolving it loses its environment.
                producer = subprocess.run(
                    [
                        str(args.sdk_python.absolute()),
                        "-c",
                        SDK_WRITE,
                        str(path),
                        store["kind"],
                        selected["runId"],
                        selected["experimentId"],
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert producer.returncode == 0, producer.stderr
                code = watcher.wait(timeout=30)
                reader.join(timeout=2)
                while not pending.empty():
                    lines.append(pending.get_nowait())
                assert code == 1, lines
                assert sum(line.startswith("run ") for line in lines) == 1
                assert any("Non-finite training loss" in line for line in lines), lines
                deletion = subprocess.run(
                    [
                        str(args.sdk_python.absolute()),
                        "-c",
                        SDK_DELETE,
                        str(path),
                        store["kind"],
                        selected["experimentId"],
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert deletion.returncode == 0, deletion.stderr
                try:
                    load_run(path)
                except ValueError as exc:
                    assert "no active runs" in str(exc)
                else:
                    raise AssertionError("deleted experiment silently selected as active")
                deleted = load_run(path, run_name=selected["runId"])
                assert deleted.meta["mlflow"]["run"]["lifecycle_stage"] == "deleted"
                transcript = "".join(lines).replace(str(path), f"capture/{store['path']}")
                name = f"{store['kind']}-watch.txt"
                (args.out / name).write_text(transcript)
                rows.append(
                    {
                        "store": store["kind"],
                        "selectedRun": selected["runId"],
                        "selectedName": selected["name"],
                        "writer": json.loads(producer.stdout),
                        "exitCode": code,
                        "deletedExperimentDefaultRefused": True,
                        "explicitDeletedRunReadable": True,
                        "transcript": name,
                    }
                )
            finally:
                if watcher.poll() is None:
                    watcher.terminate()
                    watcher.wait(timeout=5)
    (args.out / "live.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Actual CLI watch processes and native MLflow 3.17.1 writer, "
                    "in separate environments; scratch copies only."
                ),
                "python": sys.version.split()[0],
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"watchers": len(rows), "pinnedSelections": len(rows), "nativeNaNGates": len(rows)}
        )
    )


if __name__ == "__main__":
    main()
