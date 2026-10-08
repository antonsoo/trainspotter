"""Summarise the census: what the logs look like, and what each version said about them.

    uv run python studies/hub-census/report.py

Reads cache/endings.jsonl and cache/census-<label>.jsonl (see census.py) and
writes results.json and figures/alarms-{light,dark}.svg next to this file.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ENDINGS = ("good", "middling", "bad", "no progress", "short")


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["repo"]: row for row in map(json.loads, path.read_text().splitlines())}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _wilson(part: int, whole: int) -> list[float]:
    """The 95% Wilson score interval for a share, in percent."""
    if not whole:
        return [0.0, 0.0]
    z, p = 1.96, part / whole
    centre = (p + z * z / (2 * whole)) / (1 + z * z / whole)
    half = z * math.sqrt(p * (1 - p) / whole + z * z / (4 * whole * whole)) / (1 + z * z / whole)
    return [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]


def _has(row: dict[str, Any], *severities: str) -> bool:
    return any(finding[1] in severities for finding in row["findings"])


def corpus(endings: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    readable = [row for row in endings.values() if "unreadable" not in row]
    points = sorted(row["points"] for row in readable if row["points"])
    quantiles = statistics.quantiles(points, n=20)
    with_eval = [row for row in readable if row.get("eval_rise") is not None]
    rise = [row["eval_rise"] for row in with_eval]
    progressed = [row for row in readable if row.get("give_back") is not None]
    tasks = Counter(row.get("task") or "not stated" for row in readable)
    return {
        "models_scanned": manifest["models_scanned"],
        "newest": manifest["newest"],
        "oldest": manifest["oldest"],
        "per_author_cap": manifest["per_author_cap"],
        "logs": len(endings),
        "not_trainer_logs": len(endings) - len(readable),
        "readable": len(readable),
        "adapters": _pct(sum(bool(row.get("adapter")) for row in readable), len(readable)),
        "tasks": dict(tasks.most_common(6)),
        "with_train_loss": sum(row["points"] > 0 for row in readable),
        "with_eval_loss": sum(row["eval_points"] > 0 for row in readable),
        "with_grad_norm": sum("grad_norm" in row["metrics"] for row in readable),
        "train_points": {
            "p05": quantiles[0],
            "p25": quantiles[4],
            "median": quantiles[9],
            "p75": quantiles[14],
            "p95": quantiles[18],
            "max": points[-1],
        },
        "endings": dict(Counter(row["ending"] for row in readable)),
        "bad_because": dict(Counter(row["why"] for row in readable if row["ending"] == "bad")),
        "eval": {
            "runs_with_three_evaluations": len(with_eval),
            "ended_5pct_above_best": _pct(sum(r >= 0.05 for r in rise), len(rise)),
            "ended_10pct_above_best": _pct(sum(r >= 0.10 for r in rise), len(rise)),
            "ended_25pct_above_best": _pct(sum(r >= 0.25 for r in rise), len(rise)),
            "ended_50pct_above_best": _pct(sum(r >= 0.50 for r in rise), len(rise)),
        },
        "train": {
            "runs_with_clear_progress": len(progressed),
            "gave_back_10pct": _pct(
                sum(row["give_back"] >= 0.10 for row in progressed), len(progressed)
            ),
            "gave_back_25pct": _pct(
                sum(row["give_back"] >= 0.25 for row in progressed), len(progressed)
            ),
            "gave_back_50pct": _pct(
                sum(row["give_back"] >= 0.50 for row in progressed), len(progressed)
            ),
        },
    }


def verdicts(
    endings: dict[str, dict[str, Any]], findings: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = [
        {**findings[repo], "ending": ending["ending"]}
        for repo, ending in endings.items()
        if "unreadable" not in ending and "findings" in findings.get(repo, {})
    ]
    by_ending: dict[str, Any] = {}
    for name in ("all", *ENDINGS):
        group = [row for row in rows if name == "all" or row["ending"] == name]
        by_ending[name] = {
            "runs": len(group),
            "error": _pct(sum(_has(row, "error") for row in group), len(group)),
            "error_runs": sum(_has(row, "error") for row in group),
            "error_ci95": _wilson(sum(_has(row, "error") for row in group), len(group)),
            "warning_or_error": _pct(
                sum(_has(row, "error", "warning") for row in group), len(group)
            ),
            "warning_or_error_runs": sum(_has(row, "error", "warning") for row in group),
            "warning_or_error_ci95": _wilson(
                sum(_has(row, "error", "warning") for row in group), len(group)
            ),
            "any_finding": _pct(sum(bool(row["findings"]) for row in group), len(group)),
            "findings_per_run": round(
                sum(len(row["findings"]) for row in group) / max(1, len(group)), 2
            ),
        }
    errors = [row for row in rows if _has(row, "error")]
    alarms = [row for row in rows if _has(row, "error", "warning")]
    detectors: dict[str, Any] = {}
    for name in sorted({finding[0] for row in rows for finding in row["findings"]}):
        flagged = [row for row in rows if any(f[0] == name for f in row["findings"])]
        alarmed = [
            row for row in rows if any(f[0] == name and f[1] != "info" for f in row["findings"])
        ]
        detectors[name] = {
            "runs": len(flagged),
            "share": _pct(len(flagged), len(rows)),
            "runs_warning_or_error": len(alarmed),
            "findings": sum(f[0] == name for row in rows for f in row["findings"]),
        }
    return {
        "runs": len(rows),
        "by_ending": by_ending,
        "errors": {
            "runs": len(errors),
            "on_bad_endings": _pct(sum(row["ending"] == "bad" for row in errors), len(errors)),
            "on_good_endings": _pct(sum(row["ending"] == "good" for row in errors), len(errors)),
        },
        "alarms": {
            "runs": len(alarms),
            "on_bad_endings": _pct(sum(row["ending"] == "bad" for row in alarms), len(alarms)),
            "on_good_endings": _pct(sum(row["ending"] == "good" for row in alarms), len(alarms)),
        },
        "detectors": detectors,
    }


# ---------------------------------------------------------------------------
# One figure: how often each version raised an alarm, by how the run ended.
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "before": "#898781",
        "after": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "before": "#898781",
        "after": "#3987e5",
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
BAR = 22  # bar width in pixels


def alarm_figure(
    before: dict[str, Any],
    after: dict[str, Any],
    labels: tuple[str, str],
    theme: str,
    slice_name: str = "",
) -> str:
    c = THEMES[theme]
    width, height = 880, 372
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" font-family="{FONT}">',
        "<title>Share of public training runs each version flagged, by how the run ended</title>",
        f'<rect width="{width}" height="{height}" fill="{c["surface"]}"/>',
    ]

    def text(
        x: float, y: float, content: str, size: float = 12, fill: str = "secondary", **extra: str
    ) -> None:
        attributes = "".join(
            f' {name.replace("_", "-")}="{value}"' for name, value in extra.items()
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{c[fill]}"{attributes}>'
            f"{escape(content)}</text>"
        )

    def line(x1: float, y1: float, x2: float, y2: float, stroke: str) -> None:
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c[stroke]}"/>'
        )

    def bar(x: float, top_y: float, base_y: float, fill: str, tooltip: str) -> None:
        # Rounded at the data end, square on the baseline.
        r = min(4.0, base_y - top_y)
        end = x + BAR
        path = (
            f"M{x:.1f},{base_y} V{top_y + r:.1f} Q{x:.1f},{top_y:.1f} {x + r:.1f},{top_y:.1f} "
            f"H{end - r:.1f} Q{end:.1f},{top_y:.1f} {end:.1f},{top_y + r:.1f} V{base_y} Z"
        )
        parts.append(f'<path d="{path}" fill="{c[fill]}"><title>{escape(tooltip)}</title></path>')

    text(
        24,
        30,
        "How often a run was flagged, by how it actually ended",
        16,
        "ink",
        font_weight="600",
    )
    judged = sum(before["by_ending"][ending]["runs"] for ending in ("good", "middling", "bad"))
    text(
        24,
        50,
        f"Share of runs with at least one finding of that severity, among {judged:,} public "
        f"Trainer runs whose ending could be judged{slice_name}.",
        12.5,
    )
    x = 24.0
    for key, label in zip(("before", "after"), labels, strict=True):
        parts.append(f'<rect x="{x:.0f}" y="68" width="11" height="11" rx="2" fill="{c[key]}"/>')
        text(x + 17, 78, label, 12.5)
        x += 17 + 6.6 * len(label) + 26

    panels = [("error", "an error"), ("warning_or_error", "a warning or an error")]
    groups = [("good", "ended at its best"), ("middling", "in between"), ("bad", "ended badly")]
    top, bottom = 126, height - 46
    panel_width = (width - 24 - 24 - 40) / 2
    for p, (measure, title) in enumerate(panels):
        left = 24 + 34 + p * (panel_width + 40)
        right = left + panel_width - 34
        text(left, top - 20, title, 12.5, "ink")
        for tick in (0, 25, 50, 75, 100):
            y = bottom - tick / 100 * (bottom - top)
            line(left, y, right, y, "grid")
            text(left - 8, y + 4, f"{tick}%", 11.5, "muted", text_anchor="end")
        line(left, bottom, right, bottom, "axis")
        slot = (right - left) / len(groups)
        for g, (ending, caption) in enumerate(groups):
            centre = left + slot * (g + 0.5)
            for b, (key, source) in enumerate((("before", before), ("after", after))):
                value = source["by_ending"][ending][measure]
                bx = centre - BAR - 2 + b * (BAR + 4)
                by = bottom - value / 100 * (bottom - top)
                bar(bx, by, bottom, key, f"{caption}: {value:.1f}% ({labels[b]})")
                text(bx + BAR / 2, by - 6, f"{value:.0f}%", 11.5, text_anchor="middle")
            text(centre, bottom + 18, caption, 11.5, text_anchor="middle")
            count = f"{before['by_ending'][ending]['runs']:,} runs"
            text(centre, bottom + 33, count, 11, "muted", text_anchor="middle")
    return "\n".join([*parts, "</svg>"]) + "\n"


def injected(cache: Path, prefix: str, label: str, labels: tuple[str, ...]) -> dict[str, Any]:
    """For each fault, how often a version raised an alarm on runs it was quiet on before.

    A warning-or-error rate is taken over the runs the version gave no warning and no error
    unmodified; an error rate over the runs it gave no error unmodified.
    """

    def read(name: str) -> dict[str, dict[str, Any]]:
        # A second process may have worked from the other end of the list (inject.py --reverse).
        found: dict[str, dict[str, Any]] = {}
        for path in sorted(cache.glob(f"{prefix}inject-{name}*.jsonl")):
            for row in map(json.loads, path.read_text().splitlines()):
                found[row["repo"]] = row
        return found

    # Both versions are scored on the runs both have finished.
    shared = set.intersection(*(set(read(name)) for name in labels))
    rows = [row for repo, row in sorted(read(label).items()) if repo in shared]
    faults = [name for name in rows[0]["runs"] if name != "baseline"]
    # eval_* faults exist only for runs with three evaluations.
    table: dict[str, Any] = {}
    for fault in faults:
        alarm = error = alarm_n = error_n = any_alarm = any_error = total = 0
        for row in rows:
            base, hit = row["runs"]["baseline"], row["runs"].get(fault)
            if hit is None or "unreadable" in hit or "unreadable" in base:
                continue
            total += 1
            any_alarm += bool(hit["warning"] or hit["error"])
            any_error += bool(hit["error"])
            if not base["warning"] and not base["error"]:
                alarm_n += 1
                alarm += bool(hit["warning"] or hit["error"])
            if not base["error"]:
                error_n += 1
                error += bool(hit["error"])
        table[fault] = {
            "warning_or_error": _pct(alarm, alarm_n),
            "warning_or_error_of": alarm_n,
            "error": _pct(error, error_n),
            "error_of": error_n,
            # Over every damaged run, including those the version already alarmed on unmodified.
            "warning_or_error_any": _pct(any_alarm, total),
            "error_any": _pct(any_error, total),
            "of": total,
        }
    base = [row["runs"]["baseline"] for row in rows if "unreadable" not in row["runs"]["baseline"]]
    return {
        "runs": len(rows),
        "unmodified_warning_or_error": _pct(
            sum(bool(b["warning"] or b["error"]) for b in base), len(base)
        ),
        "unmodified_error": _pct(sum(bool(b["error"]) for b in base), len(base)),
        "faults": table,
    }


def summarise(
    cache: Path, prefix: str, manifest_path: Path, before: str, after: str
) -> dict[str, Any]:
    endings = _rows(cache / f"{prefix}endings.jsonl")
    manifest = json.loads(manifest_path.read_text())
    return {
        "corpus": corpus(endings, manifest),
        "before": {
            "label": before,
            **verdicts(endings, _rows(cache / f"{prefix}census-{before}.jsonl")),
        },
        "after": {
            "label": after,
            **verdicts(endings, _rows(cache / f"{prefix}census-{after}.jsonl")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--before", default="0.2.6", help="label of the earlier version's census")
    parser.add_argument("--after", default="current", help="label of the later version's census")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    args = parser.parse_args()

    results = summarise(args.cache, "", HERE / "manifest.json", args.before, args.after)
    held_path = HERE / "manifest-heldout.json"
    if held_path.exists() and (args.cache / "heldout-endings.jsonl").exists():
        held = summarise(args.cache, "heldout-", held_path, args.before, args.after)
        for label, key in ((args.before, "before"), (args.after, "after")):
            if all(
                (args.cache / f"heldout-inject-{name}.jsonl").exists()
                for name in (args.before, args.after)
            ):
                held[key]["injected"] = injected(
                    args.cache, "heldout-", label, (args.before, args.after)
                )
        results["heldout"] = held
    probe = args.cache / "probe.json"
    if probe.exists():
        results["excursion_probe"] = json.loads(probe.read_text())
    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    (HERE / "figures").mkdir(exist_ok=True)
    labels = (f"trainspotter {args.before}", "this checkout")
    for theme in THEMES:
        (HERE / "figures" / f"alarms-{theme}.svg").write_text(
            alarm_figure(results["before"], results["after"], labels, theme)
        )
        if "heldout" in results:
            (HERE / "figures" / f"alarms-heldout-{theme}.svg").write_text(
                alarm_figure(
                    results["heldout"]["before"],
                    results["heldout"]["after"],
                    labels,
                    theme,
                    " (the held-out slice)",
                )
            )

    for name in ("", "heldout"):
        part = results if not name else results.get("heldout")
        if part is None:
            continue
        print(f"== {name or 'in-sample'}")
        print(f"{part['corpus']['readable']} Trainer logs; endings {part['corpus']['endings']}")
        for key in ("before", "after"):
            r = part[key]
            print(f"-- {r['label']}")
            for ending, row in r["by_ending"].items():
                print(
                    f"   {ending:12s} {row['runs']:5d} runs  error {row['error']:5.1f}%  "
                    f"warning+ {row['warning_or_error']:5.1f}%  any {row['any_finding']:5.1f}%  "
                    f"findings/run {row['findings_per_run']:6.2f}"
                )
            print(f"   errors: {r['errors']}")


if __name__ == "__main__":
    main()
