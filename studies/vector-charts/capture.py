"""Render actual retained logs and explicit chart controls with an installed artifact."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from trainspotter.detectors import run_all
from trainspotter.model import Run
from trainspotter.readers import load_run
from trainspotter.report import render_html

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[2])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = []


def emit(name: str, run: Run, scope: str, *, detect: bool = True,
         source: Path | None = None) -> None:
    findings = run_all(run) if detect else []
    document = render_html(run, findings)
    (args.output / f"{name}.html").write_text(document)
    charts = re.findall(r'<svg class="chart".*?</svg>', document, re.S)
    exports = []
    for encoded in re.findall(r'href="data:image/svg\+xml;base64,([^"]+)"', document):
        data = base64.b64decode(encoded, validate=True)
        root = ET.fromstring(data)
        metadata = root.find("{http://www.w3.org/2000/svg}metadata")
        assert metadata is not None and metadata.text
        exports.append({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "metadata": json.loads(metadata.text)})
    row = {"name": name, "file": f"{name}.html", "scope": scope,
           "reportBytes": len(document.encode()), "reportSha256": hashlib.sha256(document.encode()).hexdigest(),
           "chartSha256": [hashlib.sha256(chart.encode()).hexdigest() for chart in charts],
           "charts": len(charts), "exports": exports, "findings": len(findings),
           "axisKind": run.axis_kind(), "axisLabel": run.axis_label(),
           "inputComplete": run.input_complete()}
    if source is not None:
        row["source"] = {"file": str(source.relative_to(args.inputs)), "bytes": source.stat().st_size,
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    rows.append(row)


for name, file, scope in [
    ("recorded-divergence", "examples/divergence.trainer_state.json",
     "Retained controlled CPU training capture deliberately made to fail; no new training executed."),
    ("recorded-logging-axis", "examples/wandb-axes/stable-logging.csv",
     "Retained native offline W&B capture, with its training counter removed; logging events are not training steps."),
]:
    source = args.inputs / file
    run = load_run(source)
    # Keep the report's source label independent of this machine's checkout path.
    run.source_path = file
    emit(name, run, scope, source=source)

run = Run(source_format="controlled", source_path="controlled-gap.jsonl")
for step, value in enumerate([.4, None, .3, math.inf, .2]):
    run.add_point("train/loss", step, value)
emit("unreadable-gap", run, "Constructed known-step unreadable and infinite values; finite segments must remain separate.")

run = Run(source_format="controlled", source_path="controlled-contradiction.jsonl")
run.add_point("train/loss", 0, 2)
run.add_point("train/loss", 0, 3)
run.add_point("eval/loss", 0, 2)
run.add_point("eval/loss", 1, 1)
emit("partial-peer", run, "Constructed conflicting training quantity with an independently readable evaluation trajectory.")
del run.metrics["eval/loss"]
emit("no-trajectory", run, "Only the contradictory metric remains; no vector download should invent a trajectory.")

name = '<script onload="bad()"> & ]]> 雨\x00\x01'
run = Run(source_path=name, source_format="controlled", meta={"metric_sources": {name: "original ' column"}})
run.add_point(name, 0, 1)
run.add_point(name, 1, 2)
emit("unsafe-names", run, "Constructed markup-shaped names and illegal XML characters; names remain data.")

run = Run(source_format="controlled", source_path="controlled-200000-observations")
for i in range(200_000):
    value = 4.0 if i == 123_456 else 2.5 * math.exp(-i / 50_000) + .3 + .02 * math.sin(i)
    run.add_point("train/loss", i, value)
emit("long-curve", run, "Constructed 200,000 observations; rendering size and preserved one-step spike, not detector accuracy.", detect=False)

receipt = {"python": platform.python_version(), "scope": "Native retained training/logging captures plus explicit display and XML controls. No extraction accuracy, new training result or invoice claim.", "cases": rows}
(args.output / 'capture.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({"cases": len(rows), "charts": sum(row['charts'] for row in rows),
                  "exports": sum(len(row['exports']) for row in rows)}))
