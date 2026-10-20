"""Exercise the installed console script and API against an archived checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import trainspotter
from trainspotter import load_run
from trainspotter.detectors import run_all
from trainspotter.report.json_report import to_json_dict


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    package = Path(trainspotter.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve())
    source = archive / "src/trainspotter"
    modules = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))}
    assert modules == {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    assets = [
        "readers/wandb-schema.pb",
        "readers/wandb-schema.json",
        "readers/WANDB-SCHEMA-LICENSES.txt",
    ]
    assert all((source / p).read_bytes() == (package / p).read_bytes() for p in assets)
    assert importlib.util.find_spec("wandb") is None
    executable = str(Path(sys.executable).parent / "trainspotter")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    rows = []
    for case, gate in [
        ("stable", 0),
        ("unstable-lr", 1),
        ("logged-null", 2),
        ("custom-counter", 0),
        ("types", 2),
    ]:
        relative = Path("examples/wandb-native") / case / "run.wandb"
        path = archive / relative
        before = sha(path)
        run = load_run(path)
        reference = to_json_dict(run, run_all(run))
        reference["run"]["source_path"] = str(relative)
        result = subprocess.run(
            [executable, "analyze", str(relative), "--output", "json", "--fail-on", "error"],
            cwd=archive,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == gate, (case, result.stderr)
        assert json.loads(result.stdout) == reference
        protected = subprocess.run(
            [executable, "analyze", str(relative), "--output", "html", "--out", str(relative)],
            cwd=archive,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert protected.returncode == 2 and sha(path) == before
        watch = subprocess.run(
            [executable, "watch", str(relative), "--once", "--no-color"],
            cwd=archive,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert watch.returncode == 0 and "Native W&B run" in watch.stdout
        assert sha(path) == before
        rows.append(
            {
                "case": case,
                "input_sha256": before,
                "gate": gate,
                "console_json_matches_api": True,
                "input_overwrite_refused": True,
                "watch_once": "actual installed console script succeeded",
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "protobuf", "trainspotter"]
                },
                "wheel_sha256": sha(args.wheel),
                "runtime_modules": modules,
                "source_is_imported": False,
                "wandb_sdk_installed": False,
                "schema_assets": {p: sha(package / p) for p in assets},
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("Installed console/API/watch/input protection verified:", platform.python_version())


if __name__ == "__main__":
    main()
