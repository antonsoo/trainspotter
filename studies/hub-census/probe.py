"""How often a run that ended well went through a large excursion, and what each version said.

    uv run python studies/hub-census/probe.py

For every run labelled "good" (both the first census and the held-out slice), find the
excursions the `setbacks` detector itself would find (`detectors.base.levels` and
`excursions`), take the highest one's height in units of the run's progress, and count how
many of the runs with a tall excursion got a warning or an error from each version. No
detector's output is used to choose the runs. Writes cache/probe.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
HEIGHTS = (0.5, 1.0, 2.0)


def cached_path(row: dict[str, Any]) -> Path:
    key = f"{row['repo']}@{row['revision']}:{row['path']}"
    return CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def rows(path: Path) -> dict[str, dict[str, Any]]:
    return {r["repo"]: r for r in map(json.loads, path.read_text().splitlines())}


def probe(prefix: str, manifest: Path) -> dict[str, Any]:
    from trainspotter.detectors.base import excursions, levels
    from trainspotter.readers.hf import read_hf_trainer_state

    logs = {r["repo"]: r for r in json.loads(manifest.read_text())["logs"]}
    endings = rows(CACHE / f"{prefix}endings.jsonl")
    said = {label: rows(CACHE / f"{prefix}census-{label}.jsonl") for label in ("0.2.6", "current")}
    tall: list[dict[str, Any]] = []
    good = 0
    for repo, ending in endings.items():
        if ending.get("ending") != "good":
            continue
        good += 1
        series = read_hf_trainer_state(cached_path(logs[repo])).get("train/loss")
        lv = levels(np.array(series.values(), dtype=float)) if series else None
        if lv is None or not lv.has_progress:
            continue
        height = max(((e.peak - e.base) / lv.progress for e in excursions(lv)), default=0.0)
        flags = {
            label: any(f[1] in ("warning", "error") for f in said[label][repo]["findings"])
            for label in said
        }
        tall.append({"repo": repo, "height": height, **flags})
    out: dict[str, Any] = {"good_runs": good}
    for h in HEIGHTS:
        taller = [t for t in tall if t["height"] >= h]
        out[f"height_{h}"] = {
            "runs": len(taller),
            "flagged_0.2.6": sum(t["0.2.6"] for t in taller),
            "flagged_current": sum(t["current"] for t in taller),
            "silent_in_current": [t["repo"] for t in taller if not t["current"]],
        }
    return out


def main() -> None:
    result = {
        "in_sample": probe("", HERE / "manifest.json"),
        "heldout": probe("heldout-", HERE / "manifest-heldout.json"),
    }
    (CACHE / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: {h: v for h, v in d.items() if h != "silent_in_current"}
                for k, d in result.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
