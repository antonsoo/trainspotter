"""Compare complete diagnostic JSON on cached, revision-pinned public training logs.

Uses separate interpreters for the two source trees. No downloads, no relabeling
of the old corpus, and no detector-derived oracle: this is a compatibility replay.
Numerical accuracy is checked independently in tests/test_rolling_statistics.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(source: Path, manifests: list[Path], cache: Path, out: Path) -> None:
    sys.path.insert(0, str(source.resolve()))
    from trainspotter.detectors import run_all
    from trainspotter.readers import load_run
    from trainspotter.report.json_report import to_json_dict

    seen = set()
    with out.open("w") as handle:
        for manifest in manifests:
            for row in json.loads(manifest.read_text())["logs"]:
                identity = f"{row['repo']}@{row['revision']}:{row['path']}"
                if identity in seen:
                    continue
                seen.add(identity)
                path = cache / (hashlib.sha1(identity.encode()).hexdigest() + ".json")
                record: dict[str, Any] = {"identity": identity}
                if not path.exists():
                    record["unavailable"] = "not in local cache"
                else:
                    record["sha256"] = digest(path)
                    try:
                        run = load_run(path, "hf")
                    except (ValueError, OSError) as exc:
                        record["unreadable"] = str(exc).replace(str(cache), "<cache>")
                    else:
                        report = to_json_dict(run, run_all(run))
                        report["run"]["source_path"] = path.name
                        record["report"] = report
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                if len(seen) % 200 == 0:
                    handle.flush()
                    print(f"{out.stem}: {len(seen)} logs", flush=True)


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for key in a for d in differences(a[key], b[key], f"{path}/{key}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (v, w) in enumerate(zip(a, b, strict=True))
            for d in differences(v, w, f"{path}/{i}")
        ]
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    ):
        return []
    return [] if a == b else [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument("--cache", type=Path, default=ROOT / "studies/hub-census/cache")
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifests = args.manifest or [
        ROOT / "studies/hub-census" / name for name in ["manifest.json", "manifest-heldout.json"]
    ]
    if args.worker:
        worker(args.source, manifests, args.cache, args.out)
        return
    if args.baseline_source is None or args.artifacts is None:
        parser.error("--baseline-source and --artifacts are required")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, source in [("before", args.baseline_source), ("after", args.source)]:
        target = args.artifacts / f"{label}.jsonl"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(source),
            "--cache",
            str(args.cache),
            "--out",
            str(target),
        ]
        for manifest in manifests:
            command.extend(["--manifest", str(manifest)])
        started = time.perf_counter()
        subprocess.run(command, check=True)
        runs.append(
            {
                "label": label,
                "elapsed_s": time.perf_counter() - started,
                "output_sha256": digest(target),
                "source_sha256": {
                    str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))
                },
            }
        )
    before = [
        json.loads(line) for line in (args.artifacts / "before.jsonl").read_text().splitlines()
    ]
    after = [json.loads(line) for line in (args.artifacts / "after.jsonl").read_text().splitlines()]
    changed, rounded, exact, readable, unavailable = [], 0, 0, 0, 0
    for a, b in zip(before, after, strict=True):
        assert a["identity"] == b["identity"]
        readable += "report" in a
        unavailable += "unavailable" in a
        diff = differences(a, b)
        if diff:
            changed.append({"identity": a["identity"], "paths": diff})
        elif a == b:
            exact += 1
        else:
            rounded += 1
    result = {
        "python": platform.python_version(),
        "manifests": {p.name: digest(p) for p in manifests},
        "runs": runs,
        "compared": len(before),
        "readable": readable,
        "unavailable": unavailable,
        "unreadable": len(before) - readable - unavailable,
        "exactly_equal": exact,
        "only_float_roundoff": rounded,
        "float_comparison": {"relative_tolerance": 1e-10, "absolute_tolerance": 1e-12},
        "changed": changed,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
