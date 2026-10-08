"""Run trainspotter over the collected logs and record how each run ended.

    uv run python studies/hub-census/census.py                       # this checkout
    uv run --no-project --with trainspotter==0.2.6 --with numpy \\
        python studies/hub-census/census.py --label 0.2.6           # a released version

Each log gets two independent descriptions:

- what the detectors said (one row per finding: detector, severity), and
- how the run ended, from the loss values alone: "good" if it finished at its
  best level, "bad" if it gave back at least half of its progress, ended at
  least 25% above its best evaluation loss, or logged a non-finite loss.

The second does not call any detector. Rows go to cache/census-<label>.jsonl;
`--endings-only` writes just the second description to cache/endings.jsonl, so
several versions' findings can be judged against one set of endings.
`report.py` joins and summarises them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
# Below this many logged training-loss points there is no level to speak of.
MIN_POINTS = 30
GOOD_GIVE_BACK, BAD_GIVE_BACK = 0.10, 0.50
GOOD_EVAL_RISE, BAD_EVAL_RISE = 0.05, 0.25


def cached_path(cache: Path, row: dict[str, Any]) -> Path:
    key = f"{row['repo']}@{row['revision']}:{row['path']}"
    return cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    return np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]
    )


def ending(train: list[float], evaluation: list[float]) -> dict[str, Any]:
    """Describe how a run ended, from its loss values alone."""
    out: dict[str, Any] = {"points": len(train), "eval_points": len(evaluation)}
    finite_eval = [v for v in evaluation if math.isfinite(v)]
    if len(finite_eval) >= 3 and min(finite_eval) > 0:
        out["eval_rise"] = finite_eval[-1] / min(finite_eval) - 1.0
    eval_rise = out.get("eval_rise")
    if len(train) < MIN_POINTS:
        return {**out, "ending": "short"}
    values = np.array(train, dtype=float)
    if not np.all(np.isfinite(values)):
        return {**out, "ending": "bad", "why": "non-finite loss"}
    edge = max(3, len(values) // 20)
    smooth = rolling_median(values, edge | 1)
    # Many runs lose most of their loss within the first few logged points, so the
    # start is the higher of the first three points and the first 5%.
    first_points, first_share = float(np.median(values[:3])), float(np.median(values[:edge]))
    start, start_count = (first_points, 3) if first_points > first_share else (first_share, edge)
    final = float(np.median(values[-edge:]))
    best = float(smooth.min())
    progress = start - best
    # How far a median of k points strays from the level by chance.
    # From successive differences: a residual to a three-point median understates it.
    change = np.diff(values)
    scatter = 1.4826 * float(np.median(np.abs(change - np.median(change)))) / math.sqrt(2)
    level_noise = 1.2533 * scatter / math.sqrt(edge)
    if values.min() > 0:
        log_ratio = np.log(values / smooth)
        relative = 1.4826 * float(np.median(np.abs(log_ratio - np.median(log_ratio))))
        scatter_at_start = max(scatter, relative * start)
    else:
        scatter_at_start = scatter
    start_noise = 1.2533 * scatter_at_start / math.sqrt(start_count)
    out.update(start=start, best=best, final=final)
    # A loss that never clearly went below where it started is not a curve these labels
    # describe: a reward-driven objective, a run that learned nothing, a flat noisy line.
    if progress <= max(6.0 * level_noise, 4.0 * start_noise):
        if (
            start > 0
            and final >= 1.5 * start
            and final - start >= max(6.0 * level_noise, 4.0 * start_noise)
        ):
            return {**out, "ending": "bad", "why": "ended half again above where it started"}
        return {**out, "ending": "no progress"}
    give_back = (final - best) / progress
    out["give_back"] = give_back
    if give_back >= BAD_GIVE_BACK and final - best >= 4.0 * level_noise:
        return {**out, "ending": "bad", "why": "gave back half its progress"}
    if eval_rise is not None and eval_rise >= BAD_EVAL_RISE:
        return {**out, "ending": "bad", "why": "evaluation loss 25% above its best"}
    if give_back <= GOOD_GIVE_BACK and (eval_rise is None or eval_rise <= GOOD_EVAL_RISE):
        return {**out, "ending": "good"}
    return {**out, "ending": "middling"}


def main() -> None:
    import trainspotter
    from trainspotter.detectors import run_all
    from trainspotter.readers.hf import read_hf_trainer_state

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--label", default="current", help="names the output file")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument(
        "--endings-only",
        action="store_true",
        help="write cache/endings.jsonl (how each run ended) without running any detector",
    )
    parser.add_argument(
        "--prefix", default="", help="prepended to the output file names (the held-out slice)"
    )
    args = parser.parse_args()

    logs = json.loads(args.manifest.read_text())["logs"]
    name = "endings.jsonl" if args.endings_only else f"census-{args.label}.jsonl"
    target = args.cache / (args.prefix + name)
    started = time.time()
    with target.open("w") as handle:
        for row in logs:
            record: dict[str, Any] = {
                "repo": row["repo"],
                "adapter": row["adapter"],
                "task": row["task"],
            }
            try:
                run = read_hf_trainer_state(cached_path(args.cache, row))
            except (ValueError, OSError) as exc:
                record["unreadable"] = str(exc).split(": ", 1)[-1][:120]
                handle.write(json.dumps(record) + "\n")
                continue
            train, evaluation = run.get("train/loss"), run.get("eval/loss")
            record.update(
                ending(train.values() if train else [], evaluation.values() if evaluation else [])
            )
            record["metrics"] = sorted(run.metric_names())
            if not args.endings_only:
                record["findings"] = [[f.detector, f.severity, f.title] for f in run_all(run)]
            handle.write(json.dumps(record) + "\n")
    elapsed = time.time() - started
    print(f"trainspotter {trainspotter.__version__}: {len(logs)} logs in {elapsed:.0f} s")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
