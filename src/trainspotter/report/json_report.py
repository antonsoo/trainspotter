"""JSON report: the machine-readable form, used by `--format json` and by
the HTML report internally (it renders from the same dict this module
produces, so the two never disagree)."""

from __future__ import annotations

import json
import math
from typing import Any

from trainspotter.detectors import Finding
from trainspotter.model import Run
from trainspotter.timing import assess_timing

SCHEMA_VERSION = 1


def _json_safe(value: float | int | str) -> float | int | str:
    """A diverged run legitimately produces NaN/Inf evidence values (e.g. a
    spike's z-score against an infinite metric reading). Python's `json`
    module emits those as bare `Infinity`/`NaN` tokens, which is valid for
    `json.loads` but not standard JSON (RFC 8259) -- most other parsers,
    including JavaScript's `JSON.parse`, reject it. Stringify them instead
    so the output is portable."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _several_runs(run: Run) -> dict[str, Any]:
    """`selected_run` and `runs`, present only when the source held more than one run."""
    if run.other_runs() is None:
        return {}
    return {"selected_run": run.meta["selected_run"], "runs": run.meta["runs"]}


def notable_repeats(run: Run) -> list[dict[str, Any]]:
    """The repeated readings a person should see: everything except agreeing `epoch`
    values, which `Trainer` writes twice at every evaluated step (see `Run.input_notes`).
    The JSON report keeps all of them."""
    return [
        record
        for record in repeated_step_evidence(run)
        if record["metric"] != "epoch"
        or record["conflicting_values"]
        or record["conflicting_times"]
    ]


def repeated_step_evidence(run: Run) -> list[dict[str, Any]]:
    """Complete decoded evidence, grouping identical readings with explicit counts."""
    result: list[dict[str, Any]] = []
    for name, groups in sorted(run.repeated_steps().items()):
        for group in groups:
            readings: dict[tuple[str, str, str], dict[str, Any]] = {}
            for point in group.readings:
                key = (float(point.value).hex(), str(point.wall_time), str(point.epoch))
                if key not in readings:
                    readings[key] = {
                        "value": _json_safe(point.value),
                        "wall_time": _json_safe(point.wall_time)
                        if point.wall_time is not None
                        else None,
                        "epoch": _json_safe(point.epoch) if point.epoch is not None else None,
                        "count": 0,
                    }
                readings[key]["count"] += 1
            result.append(
                {
                    "metric": name,
                    "step": group.step,
                    "count": len(group.readings),
                    "conflicting_values": group.conflicting_values,
                    "conflicting_times": group.conflicting_times,
                    "readings": list(readings.values()),
                }
            )
    return result


def to_json_dict(run: Run, findings: list[Finding]) -> dict[str, Any]:
    analysis = run.analysis_view()
    by_severity = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        by_severity[f.severity] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "source_format": run.source_format,
            "source_path": run.source_path,
            "skipped_rows": run.skipped_rows(),
            "skipped_values": run.skipped_values(),
            "input_complete": run.input_complete(),
            "input_notes": run.input_notes(),
            "unread_event_bytes": run.unread_event_bytes(),
            "overwritten_points": run.overwritten_points(),
            "restart_markers": run.meta.get("restart_markers", 0),
            "step_column": run.meta.get("step_column", "step"),
            "step_axis": run.axis_kind(),
            "step_selection": run.meta.get("step_selection", "automatic"),
            "metric_sources": run.meta.get("metric_sources", {}),
            "ambiguous_metric_sources": run.meta.get("ambiguous_metric_sources", []),
            "repeated_steps": repeated_step_evidence(run),
            **_several_runs(run),
            "metrics": {
                name: {
                    "count": len(series),
                    "analysis_count": len(analysis.metrics[name])
                    if name in analysis.metrics
                    else 0,
                    "distinct_steps": len({point.step for point in series.points}),
                    "analysis_excluded": bool(series.points) and name not in analysis.metrics,
                    "first_step": series.steps()[0] if not series.is_empty() else None,
                    "last_step": series.steps()[-1] if not series.is_empty() else None,
                }
                for name, series in sorted(run.metrics.items())
            },
        },
        "summary": {
            "input_complete": run.input_complete(),
            "total_findings": len(findings),
            "by_severity": by_severity,
        },
        "timing": assess_timing(run).to_dict(),
        "findings": [
            {
                "detector": f.detector,
                "title": f.title,
                "severity": f.severity,
                "metric": f.metric,
                "step_start": f.step_start,
                "step_end": f.step_end,
                "message": f.message,
                "evidence": {k: _json_safe(v) for k, v in f.evidence.items()},
                "fixes": f.fixes,
            }
            for f in findings
        ],
    }


def to_json_str(run: Run, findings: list[Finding], indent: int = 2) -> str:
    # allow_nan=False: fail loudly if a non-finite value ever reaches here
    # unstringified, rather than silently emit non-standard JSON.
    return json.dumps(to_json_dict(run, findings), indent=indent, sort_keys=False, allow_nan=False)
