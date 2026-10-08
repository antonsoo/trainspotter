"""Inject faults of known size into held-out logs that ended well, and see what each version says.

    uv run python studies/hub-census/inject.py --label current
    uv run --no-project --with trainspotter==0.2.6 --with numpy \\
        python studies/hub-census/inject.py --label 0.2.6

The census asks how often a version stays quiet on runs that ended well. That says nothing
about what it no longer catches. This takes the held-out runs the census labelled "good" and
damages a copy of each in a way whose size is set by a fixed rule, in units of the run's own
progress P (start level minus best level, as `census.py` computes it). Nothing here looks at
what a detector says before choosing the fault.

Faults, all applied to the logged training loss unless noted:

- `nan_mid`, `nan_last`: one logged value replaced by NaN, at the middle / at the end.
- `late_<g>`: the last 10% of the run climbs linearly by g*P and stays there, so the run
  ends having given back about g of its progress (g = 0.25, 0.5, 1.0).
- `bump_<k>`: between 40% and 55% of the run the level is raised by k*P (a ramp up over 3% of
  the run, a plateau, a ramp down over 3%) and then comes back (k = 0.25, 0.5, 1, 2, 4).
- `burst_<k>`: a short excursion: 2% of the run (at least three logged values), centred on the
  middle, raised by k*P (k = 0.5, 1, 2, 4). Wider than a spike, narrower than a setback.
- `spike_<k>`: one logged value at the middle raised by k*P (k = 0.5, 1, 2, 4).
- `eval_last_<r>`: the last evaluation loss set to (1 + r) times the lowest earlier one, for
  runs with at least three evaluations (r = 0.05, 0.1, 0.25, 0.5, 1.0).
- `eval_ramp_<r>`: the same final level, reached by a steady climb over the last quarter of the
  evaluations (at least two), each at least as high as it really was.

The runs are the first MAX_RUNS "good" runs with 30 to 3,000 logged training-loss points,
ordered by the SHA-1 of the repository name. Output: cache/heldout-inject-<label>.jsonl.
`--workers N` shares the list between N processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
MAX_RUNS = 300
MAX_POINTS = 3000

LATE = (0.25, 0.5, 1.0)
BUMP = (0.25, 0.5, 1.0, 2.0, 4.0)
SPIKE = (0.5, 1.0, 2.0, 4.0)
EVAL = (0.05, 0.1, 0.25, 0.5, 1.0)


def cached_path(cache: Path, row: dict[str, Any]) -> Path:
    key = f"{row['repo']}@{row['revision']}:{row['path']}"
    return cache / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def variants(history: list[dict[str, Any]], progress: float) -> dict[str, list[dict[str, Any]]]:
    train = [i for i, e in enumerate(history) if isinstance(e.get("loss"), (int, float))]
    evals = [i for i, e in enumerate(history) if isinstance(e.get("eval_loss"), (int, float))]
    n = len(train)
    out: dict[str, list[dict[str, Any]]] = {"baseline": history}

    def edited(change: dict[int, float], key: str = "loss") -> list[dict[str, Any]]:
        copy = [dict(e) for e in history]
        for i, value in change.items():
            copy[i][key] = value
        return copy

    def shifted(offsets: np.ndarray) -> list[dict[str, Any]]:
        return edited({train[j]: history[train[j]]["loss"] + float(offsets[j]) for j in range(n)})

    out["nan_mid"] = edited({train[n // 2]: math.nan})
    out["nan_last"] = edited({train[-1]: math.nan})
    position = np.arange(n) / n
    for g in LATE:
        ramp = np.clip((position - 0.90) / 0.05, 0.0, 1.0)
        out[f"late_{g}"] = shifted(g * progress * ramp)
    for k in BUMP:
        up = np.clip((position - 0.40) / 0.03, 0.0, 1.0)
        down = np.clip((0.55 - position) / 0.03, 0.0, 1.0)
        out[f"bump_{k}"] = shifted(k * progress * np.minimum(up, down))
    width = max(3, round(0.02 * n))
    for k in SPIKE:
        block = np.zeros(n)
        block[n // 2 - width // 2 : n // 2 - width // 2 + width] = k * progress
        out[f"burst_{k}"] = shifted(block)
    for k in SPIKE:
        point = np.zeros(n)
        point[n // 2] = k * progress
        out[f"spike_{k}"] = shifted(point)
    values = [history[i]["eval_loss"] for i in evals]
    m = max(2, round(len(evals) / 4))
    # The size r is measured from the lowest evaluation before the ones that are damaged, so a
    # run whose real last evaluation was its best is not understated.
    if len(evals) >= 3 and all(math.isfinite(v) for v in values):
        for r in EVAL:
            for name, kept in (("last", values[:-1]), ("ramp", values[:-m])):
                low = min(kept)
                if low <= 0:
                    continue
                # The tiny factor keeps a rise of exactly r from landing a rounding error below it.
                final = low * (1.0 + r) * (1.0 + 1e-9)
                change = {evals[-1]: final}
                if name == "ramp":
                    for j in range(m - 1):
                        index = evals[-m + j]
                        change[index] = max(
                            values[len(evals) - m + j], low * (1.0 + r * (j + 1) / m)
                        )
                out[f"eval_{name}_{r}"] = edited(change, "eval_loss")
    return out


def judge(job: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """Damage one run every way `variants` knows and record what the detectors say of each."""
    from trainspotter.detectors import run_all
    from trainspotter.readers.hf import read_hf_trainer_state

    e, row, cache = job
    raw = json.loads(cached_path(Path(cache), row).read_text())
    history = raw["log_history"] if isinstance(raw, dict) else raw
    record: dict[str, Any] = {"repo": e["repo"], "progress": e["start"] - e["best"], "runs": {}}
    with tempfile.TemporaryDirectory() as scratch:
        for name, damaged in variants(history, e["start"] - e["best"]).items():
            path = Path(scratch) / "trainer_state.json"
            path.write_text(json.dumps({"log_history": damaged}))
            try:
                findings = run_all(read_hf_trainer_state(path))
            except ValueError as exc:
                record["runs"][name] = {"unreadable": str(exc)[:100]}
                continue
            record["runs"][name] = {
                "warning": sorted({f.detector for f in findings if f.severity == "warning"}),
                "error": sorted({f.detector for f in findings if f.severity == "error"}),
            }
    return record


def main() -> None:
    import trainspotter

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--label", default="current", help="names the output file")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest-heldout.json")
    parser.add_argument("--prefix", default="heldout-")
    parser.add_argument("--workers", type=int, default=1, help="processes to share the runs")
    args = parser.parse_args()

    logs = {row["repo"]: row for row in json.loads(args.manifest.read_text())["logs"]}
    endings = [
        json.loads(line)
        for line in (args.cache / f"{args.prefix}endings.jsonl").read_text().splitlines()
    ]
    chosen = [e for e in endings if e.get("ending") == "good" and 30 <= e["points"] <= MAX_POINTS]
    chosen.sort(key=lambda e: hashlib.sha1(e["repo"].encode()).hexdigest())
    chosen = chosen[:MAX_RUNS]

    target = args.cache / f"{args.prefix}inject-{args.label}.jsonl"
    jobs = [(e, logs[e["repo"]], str(args.cache)) for e in chosen]
    with target.open("w") as handle, Pool(args.workers) as pool:
        for done, record in enumerate(pool.imap(judge, jobs), start=1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if done % 25 == 0:
                print(f"{done}/{len(chosen)}", flush=True)
    print(f"trainspotter {trainspotter.__version__}: {len(chosen)} runs; wrote {target}")


if __name__ == "__main__":
    main()
