"""Measure CSV loading with ten percent rejected rows at increasing input sizes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = """
import json,platform,statistics,time
from pathlib import Path
from trainspotter.readers import load_run
records=[]
for count in (1000,10000,100000):
    path=Path('scaling.csv')
    path.write_text('step,loss,comment\\n'+''.join(
        f'{i}\\n' if i%10==5 else f'{i},1.0,ok\\n' for i in range(count)))
    samples=[]
    for _ in range(3):
        start=time.perf_counter()
        run=load_run(path)
        samples.append(time.perf_counter()-start)
    series=run.get('train/loss')
    assert series.readable_count()==count*9//10 and run.skipped_rows()==count//10
    records.append({'rows':count,'bytes':path.stat().st_size,'rejectedRows':run.skipped_rows(),
                    'unreadablePositions':sum(p.value is None for p in series.points),
                    'seconds':samples,'medianSeconds':statistics.median(samples)})
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),
 'machine':'14 vCPU WSL2 Linux, 48 GB RAM','scope':'load_run only; three samples per size',
 'records':records}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve() / "src")
    with tempfile.TemporaryDirectory(prefix="csv-row-scaling-", dir="/tmp/claude-1000") as work:
        result = subprocess.run(
            [str(args.python.absolute()), "-c", PROBE],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    payload = json.loads(result.stdout)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print([(r["rows"], round(r["medianSeconds"], 4)) for r in payload["records"]])


if __name__ == "__main__":
    main()
