"""Build real-example reports and explicitly synthetic adverse inputs for browser checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report import render_html, to_json_str

root = Path(__file__).resolve().parent.parent
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
sources = sorted((root / "examples").glob("*.trainer_state.json"))
partial = destination / "synthetic-partial.jsonl"
partial.write_text('{"step":100,"loss":2}\n{broken\n{"step":200,"loss":1}\n')
sources.append(partial)
hostile = destination / "synthetic-names.jsonl"
hostile.write_text("\n".join(json.dumps({
    "step": i, "loss": 2.0 - i * 0.01,
    '<img src="https://example.invalid/leak" onerror="window.injected=1">': i,
    "eval_accuracy": 0.5, "eval/accuracy": 0.9,
    "long-metric-" + "x" * 200: 1,
}) for i in range(50)))
sources.append(hostile)
ordinal = destination / "synthetic-ordinal.csv"
ordinal.write_text("loss\n2\n1\n")
sources.append(ordinal)
manifest = []
for source in sources:
    run = load_run(source)
    run.source_path = source.name
    findings = run_all(run)
    path = destination / (source.stem + ".html")
    path.write_text(render_html(run, findings), encoding="utf-8")
    path.with_suffix(".json").write_text(to_json_str(run, findings), encoding="utf-8")
    manifest.append({"file": path.name, "complete": run.input_complete(),
                     "metrics": len(run.metrics), "findings": len(findings)})
print(json.dumps(manifest))
