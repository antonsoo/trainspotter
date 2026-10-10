"""Exercise a wheel's actual CLI, outside the checkout, on all retained inputs.

Run with the wheel environment's Python:
    python /path/to/studies/diagnostic-evidence/installed.py --output DIR
This verifier does not import trainspotter or add its source tree to sys.path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reject_constant(value):
    raise ValueError(f"nonstandard JSON number: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / "studies/diagnostic-evidence/after.json").read_text())["reports"]
    executable = Path(sys.executable).parent / "trainspotter"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    results = []
    for name, reference in expected.items():
        source = ROOT / "examples/diagnostic-evidence" / f"{name}.csv"
        command = [str(executable), "analyze", str(source)]
        report_path = args.output / f"{name}.html"
        calls = [
            ("json", [*command, "--output", "json"], 0),
            ("gate", [*command, "--output", "json", "--fail-on", "warning"],
             int(any(f["severity"] in ("warning", "error") for f in reference["findings"]))),
            ("html", [*command, "--output", "html", "--out", str(report_path)], 0),
        ]
        codes = {}
        for kind, arguments, expected_exit in calls:
            result = subprocess.run(arguments, cwd=args.output, env=env, capture_output=True,
                                    text=True, timeout=30)
            assert result.returncode == expected_exit, (name, kind, result.stderr)
            assert result.stderr == "", (name, kind, result.stderr)
            codes[kind] = result.returncode
            if kind != "html":
                report = json.loads(result.stdout, parse_constant=reject_constant)
                report["run"]["source_path"] = source.name
                assert report == reference, (name, kind)
            else:
                markup = report_path.read_text()
                assert '<meta http-equiv="Content-Security-Policy"' in markup
                assert "<script" not in markup
                for finding in reference["findings"]:
                    assert finding["title"] in markup
        results.append({"input": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "exit_codes": codes, "findings": reference["summary"]["by_severity"]})
    # A one-shot watch must expose pointwise evidence even in a short history.
    watch = subprocess.run([str(executable), "watch", str(ROOT / "examples/diagnostic-evidence"
                           / "overflow-one-step.csv"), "--once", "--no-color"], cwd=args.output,
                           env=env, capture_output=True, text=True, timeout=30)
    assert watch.returncode == 0 and "Non-finite gradient norm" in watch.stdout
    result = {"python": platform.python_version(),
              "numpy": importlib.metadata.version("numpy"),
              "trainspotter": importlib.metadata.version("trainspotter"),
              "cli_calls": len(results) * 3 + 1, "watch_nonfinite_observed": True,
              "checks": results}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))


if __name__ == "__main__":
    main()
