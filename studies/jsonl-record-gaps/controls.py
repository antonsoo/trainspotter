"""Compare every pre-existing tracked CSV/JSONL's complete report or reader rejection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBE = """
import hashlib,json,platform,sys
from pathlib import Path
from trainspotter.readers import load_run
from trainspotter.detectors import run_all
from trainspotter.report.json_report import to_json_dict
records=[]
for filename in json.loads(sys.stdin.read()):
    data=Path(filename).read_bytes()
    record={'file':filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    try:
        run=load_run(filename)
        report=to_json_dict(run,run_all(run))
        encoded=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        record.update(reportSha256=hashlib.sha256(encoded).hexdigest(),
                      inputComplete=report['summary']['input_complete'],
                      findings=report['findings'])
    except ValueError as error:
        record['rejection']=str(error)
    records.append(record)
print(json.dumps({'python':platform.python_version(),'records':records},allow_nan=False))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    filenames = subprocess.check_output(
        ["git", "ls-files", "*.csv", "*.jsonl"],
        cwd=REPO,
        text=True,
    ).splitlines()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    result = subprocess.run(
        [str(args.python.absolute()), "-c", PROBE],
        cwd=REPO,
        env=env,
        input=json.dumps(filenames),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert len(payload["records"]) == len(filenames)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"{len(filenames)} complete CSV/JSONL reports/rejections recorded")


if __name__ == "__main__":
    main()
