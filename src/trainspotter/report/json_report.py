"""JSON report: the machine-readable form, used by `--format json` and by
the HTML report internally (it renders from the same dict this module
produces, so the two never disagree)."""

from __future__ import annotations

import json
from typing import Any

from trainspotter.detectors import Finding
from trainspotter.model import Run

SCHEMA_VERSION = 1


def to_json_dict(run: Run, findings: list[Finding]) -> dict[str, Any]:
    by_severity = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        by_severity[f.severity] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "source_format": run.source_format,
            "source_path": run.source_path,
            "metrics": {
                name: {
                    "count": len(series),
                    "first_step": series.steps()[0] if not series.is_empty() else None,
                    "last_step": series.steps()[-1] if not series.is_empty() else None,
                }
                for name, series in sorted(run.metrics.items())
            },
        },
        "summary": {
            "total_findings": len(findings),
            "by_severity": by_severity,
        },
        "findings": [
            {
                "detector": f.detector,
                "title": f.title,
                "severity": f.severity,
                "metric": f.metric,
                "step_start": f.step_start,
                "step_end": f.step_end,
                "message": f.message,
                "evidence": f.evidence,
                "fixes": f.fixes,
            }
            for f in findings
        ],
    }


def to_json_str(run: Run, findings: list[Finding], indent: int = 2) -> str:
    return json.dumps(to_json_dict(run, findings), indent=indent, sort_keys=False)
