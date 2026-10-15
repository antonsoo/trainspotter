"""Compare runtime bytes in source, sdist, wheel and two isolated installations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--minimum", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    runtime = source / "src/trainspotter"
    expected = {str(p.relative_to(runtime)): sha(p.read_bytes())
                for p in sorted(runtime.rglob("*"))
                if p.is_file() and p.suffix in {".py", ".typed"}}
    wheel, = (source / "dist").glob("*.whl")
    sdist, = (source / "dist").glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        actual = {name.removeprefix("trainspotter/"): sha(archive.read(name))
                  for name in archive.namelist() if name.startswith("trainspotter/")}
    assert actual == expected
    with tarfile.open(sdist) as archive:
        actual = {}
        for member in archive.getmembers():
            if "/src/trainspotter/" in member.name and member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                actual[member.name.split("/src/trainspotter/", 1)[1]] = sha(stream.read())
    assert actual == expected
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    installations = []
    for python in (args.minimum, args.current):
        result = subprocess.run([str(python.absolute()), "-I", "-c", """
import hashlib,importlib.util,json,numpy,platform
from pathlib import Path
import trainspotter
root=Path(trainspotter.__file__).parent
print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,
 'tensorboardInstalled':importlib.util.find_spec('tensorboard') is not None,
 'runtimeFiles':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.typed'}}}))
"""], capture_output=True, text=True, env=env, cwd=args.reports, check=True)
        record = json.loads(result.stdout)
        assert record.pop("runtimeFiles") == expected
        assert not record["tensorboardInstalled"]
        # Check the installed console entry point as well as -m replay.
        cli = python.absolute().with_name("trainspotter")
        result = subprocess.run([str(cli), "analyze", "rise-truncated.jsonl",
                                 "--output", "json", "--fail-on", "error"],
                                capture_output=True, text=True, env=env, cwd=args.reports)
        report = json.loads(result.stdout)
        assert result.returncode == 2 and not report["summary"]["input_complete"]
        assert report["run"]["metrics"]["train/loss"]["unreadable_steps"] == list(range(50, 60))
        assert not any(f["detector"] == "divergence" for f in report["findings"])
        record.update(runtimeFilesMatched=len(expected), consoleExit=result.returncode,
                      consoleReportSha256=sha(json.dumps(report, sort_keys=True).encode()))
        installations.append(record)
    args.out.write_text(json.dumps({"runtimeFileCount": len(expected), "runtimeFiles": expected,
                                   "artifacts": [{"filename": p.name, "bytes": p.stat().st_size,
                                                  "sha256": sha(p.read_bytes())}
                                                 for p in (wheel, sdist)],
                                   "installations": installations}, indent=2) + "\n")
    print(f"{len(expected)} runtime files match source, sdist, wheel and both installations")


if __name__ == "__main__":
    main()
