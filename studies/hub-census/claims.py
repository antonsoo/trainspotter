"""Recount the figures the detectors' docstrings quote from the census.

    uv run python studies/hub-census/claims.py
    uv run --no-project --with trainspotter==0.2.6 --with numpy \\
        python studies/hub-census/claims.py --divergence

The first form needs only the cache and writes cache/claims.json. The second runs the released
0.2.6 over the logs it called diverged for a rising tail and sorts them by the metric that
rose; it adds that count to the same file. `report.py` copies the file into results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
CACHE = HERE / "cache"


def rows(path: Path) -> dict[str, dict[str, Any]]:
    return {r["repo"]: r for r in map(json.loads, path.read_text().splitlines())}


def raw(row: dict[str, Any]) -> list[dict[str, Any]]:
    key = f"{row['repo']}@{row['revision']}:{row['path']}"
    data = json.loads((CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")).read_text())
    return data["log_history"] if isinstance(data, dict) else data


def counts() -> dict[str, Any]:
    logs = {r["repo"]: r for r in json.loads((HERE / "manifest.json").read_text())["logs"]}
    endings = {k: v for k, v in rows(CACHE / "endings.jsonl").items() if "unreadable" not in v}
    said = rows(CACHE / "census-0.2.6.jsonl")
    good = [k for k, e in endings.items() if e["ending"] == "good"]
    # Runs whose training loss ended within a tenth of its progress of its best, whatever the
    # evaluation loss did: the group the divergence docstring counts.
    near_best = [
        k for k, e in endings.items() if e.get("give_back") is not None and e["give_back"] <= 0.10
    ]
    out: dict[str, Any] = {"good": len(good), "training_loss_near_best": len(near_best)}

    def has_nan(key: str, group: list[str], field: str) -> int:
        total = 0
        for k in group:
            values = [e[field] for e in raw(logs[k]) if isinstance(e.get(field), float)]
            total += any(not math.isfinite(v) for v in values)
        return total

    out["nan_eval_loss_among_near_best"] = has_nan("", near_best, "eval_loss")
    out["nan_eval_loss_among_good"] = has_nan("", good, "eval_loss")
    out["nan_grad_norm_among_good"] = has_nan("", good, "grad_norm")
    few = [
        k
        for k, e in endings.items()
        if e["eval_points"] in (3, 4) and e.get("eval_rise") is not None and e["eval_rise"] >= 0.25
    ]
    out["three_or_four_evaluations_ended_25pct_above_best"] = len(few)
    scheduled = [k for k in endings if "lr" in endings[k]["metrics"]]
    no_warmup = [
        k for k in scheduled if any(f[2] == "No LR warmup detected" for f in said[k]["findings"])
    ]
    out["runs_with_logged_schedule"] = len(scheduled)
    out["no_warmup_0.2.6"] = len(no_warmup)
    out["no_warmup_share"] = round(100 * len(no_warmup) / len(scheduled), 1)
    return out


def divergence_by_metric() -> dict[str, int]:
    from trainspotter.detectors.divergence import DivergenceDetector
    from trainspotter.readers.hf import read_hf_trainer_state

    logs = {r["repo"]: r for r in json.loads((HERE / "manifest.json").read_text())["logs"]}
    said = rows(CACHE / "census-0.2.6.jsonl")
    tally: dict[str, int] = {}
    for repo, row in said.items():
        if not any(f[2] == "Diverged (sustained growth)" for f in row.get("findings", [])):
            continue
        key = f"{repo}@{logs[repo]['revision']}:{logs[repo]['path']}"
        path = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
        run = read_hf_trainer_state(path)
        metrics = {
            f.metric
            for f in DivergenceDetector().run(run)
            if f.title == "Diverged (sustained growth)"
        }
        for metric in metrics:
            tally[metric] = tally.get(metric, 0) + 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--divergence", action="store_true", help="run under trainspotter 0.2.6")
    args = parser.parse_args()
    target = CACHE / "claims.json"
    result = json.loads(target.read_text()) if target.exists() else {}
    if args.divergence:
        result["divergence_0.2.6_rising_tail_by_metric"] = divergence_by_metric()
    else:
        result.update(counts())
    target.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
