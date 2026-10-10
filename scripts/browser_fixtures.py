"""Build real-example reports and explicitly synthetic adverse inputs for browser checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trainspotter.detectors import run_all
from trainspotter.readers import load_run
from trainspotter.report import render_html, to_json_str
from trainspotter.report.json_report import notable_repeats

root = Path(__file__).resolve().parent.parent
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
sources = sorted((root / "examples").glob("*.trainer_state.json"))
sources.extend(sorted((root / "examples/timing").glob("*.csv")))
sources.extend(sorted((root / "examples/wandb-axes").glob("*.csv")))
sources.extend(root / "examples/schedulers/captured" / f"{name}.csv" for name in (
    "linear-resume-reset", "linear-resume-restored", "cosine-overrun", "cosine-restarts",
))
sources.extend(p for p in sorted((root / "examples/evaluation/captured").glob("*.csv"))
               if p.name != "predictions.csv")
partial = destination / "synthetic-partial.jsonl"
partial.write_text('{"step":100,"loss":2}\n{broken\n{"step":200,"loss":1}\n')
sources.append(partial)
hostile = destination / "synthetic-names.jsonl"
hostile.write_text(
    "\n".join(
        json.dumps(
            {
                "step": i,
                "loss": 2.0 - i * 0.01,
                '<img src="https://example.invalid/leak" onerror="window.injected=1">': i,
                "eval_accuracy": 0.5,
                "eval/accuracy": 0.9,
                "long-metric-" + "x" * 200: 1,
            }
        )
        for i in range(50)
    )
)
sources.append(hostile)
ordinal = destination / "synthetic-ordinal.csv"
ordinal.write_text("loss\n2\n1\n")
sources.append(ordinal)
sources.extend((root / "examples/repeated_steps").iterdir())
repeated_large = destination / "synthetic-repeated-large.jsonl"
repeated_large.write_text(
    "\n".join(
        json.dumps(row)
        for step in range(150)
        for row in [
            {"step": step, "loss": 1.0, "grad_norm": 1.0},
            {"step": step, "loss": 2.0},
        ]
    )
)
sources.append(repeated_large)
manifest = []
for source in sources:
    run = load_run(source)
    run.source_path = source.name
    findings = run_all(run)
    path = destination / (source.stem + ".html")
    path.write_text(render_html(run, findings), encoding="utf-8")
    path.with_suffix(".json").write_text(to_json_str(run, findings), encoding="utf-8")
    manifest.append(
        {
            "file": path.name,
            "complete": run.input_complete(),
            "metrics": len(run.metrics),
            "findings": len(findings),
            # The section appears only when something other than agreeing `epoch`
            # duplicates repeats; when it does appear it lists every group.
            "repeatedGroups": sum(len(groups) for groups in run.repeated_steps().values())
            if notable_repeats(run)
            else 0,
        }
    )
print(json.dumps(manifest))
