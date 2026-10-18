"""Verify all runtime bytes across a clean source archive, wheel, sdist and installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


def sha(value):
    return hashlib.sha256(value).hexdigest()


def runtime(name):
    return name.endswith(".py") or name.endswith("/py.typed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("interpreters", type=Path, nargs="+")
    args = parser.parse_args()
    expected = {
        str(p.relative_to(args.source / "src")): sha(p.read_bytes())
        for p in (args.source / "src/trainspotter").rglob("*")
        if p.is_file() and runtime(str(p))
    }
    with zipfile.ZipFile(args.wheel) as wheel:
        packed = {
            name: sha(wheel.read(name))
            for name in wheel.namelist()
            if name.startswith("trainspotter/") and runtime(name)
        }
    assert packed == expected
    with tarfile.open(args.sdist) as sdist:
        unpacked = {}
        for member in sdist.getmembers():
            _, separator, name = member.name.partition("/src/")
            if separator and member.isfile() and name.startswith("trainspotter/") and runtime(name):
                unpacked[name] = sha(sdist.extractfile(member).read())
    assert unpacked == expected
    installs = []
    code = """
import hashlib, importlib.metadata, json, platform
from pathlib import Path
import trainspotter
root = Path(trainspotter.__file__).parent
files = {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in root.rglob("*") if p.is_file() and (p.suffix == ".py" or p.name == "py.typed")}
dist = importlib.metadata.distribution("trainspotter")
print(json.dumps({"python": platform.python_version(), "origin": str(root), "files": files,
                  "numpy": importlib.metadata.version("numpy"),
                  "directUrl": json.loads(dist.read_text("direct_url.json"))}))
"""
    for interpreter in args.interpreters:
        probe = subprocess.run(
            [str(interpreter.absolute()), "-I", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        actual = json.loads(probe.stdout)
        assert actual.pop("files") == expected
        assert actual["directUrl"]["url"] == args.wheel.resolve().as_uri()
        installs.append(actual)
    receipt = {
        "runtimeFiles": expected,
        "wheel": {"file": args.wheel.name, "sha256": sha(args.wheel.read_bytes())},
        "sdist": {"file": args.sdist.name, "sha256": sha(args.sdist.read_bytes())},
        "installs": installs,
        "runtimeMismatches": 0,
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"runtimeFiles": len(expected), "installs": len(installs), "mismatches": 0}))


if __name__ == "__main__":
    main()
