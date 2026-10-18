"""Compare complete CLI JSON on all previously retained native MLflow runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-python", type=Path, required=True)
    parser.add_argument("--after-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for directory in sorted((ROOT / "studies/mlflow-native/fixtures").iterdir()):
        native = json.loads(gzip.decompress((directory / "histories.json.gz").read_bytes()))
        for store in native["stores"]:
            for run in store["runs"]:
                source = (directory / store["path"]).relative_to(ROOT)
                parameters = [
                    "analyze",
                    str(source),
                    "--run",
                    run["runId"],
                    "--output",
                    "json",
                    "--fail-on",
                    "error",
                ]
                outputs = []
                for python in (args.before_python, args.after_python):
                    result = subprocess.run(
                        [str(python.absolute()), "-m", "trainspotter.cli", *parameters],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    assert result.returncode in (0, 1, 2), result.stderr
                    outputs.append((result.returncode, json.loads(result.stdout)))
                assert outputs[0] == outputs[1], (directory.name, store["kind"], run["name"])
                raw = json.dumps(outputs[1][1], sort_keys=True, allow_nan=False).encode()
                records.append(
                    {
                        "producer": directory.name,
                        "store": store["kind"],
                        "run": run["name"],
                        "runId": run["runId"],
                        "exit": outputs[1][0],
                        "completeJsonUnchanged": True,
                        "canonicalJsonSha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    args.output.write_text(
        json.dumps(
            {
                "scope": (
                    "Actual former and corrected installed CLI, complete canonical JSON and "
                    "error gate equality on all existing unqualified native MLflow controls."
                ),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(records)} complete native control reports and gate exits unchanged")


if __name__ == "__main__":
    main()
