"""TensorBoard event-file reader (optional extra: `pip install trainspotter[tensorboard]`).

Reads scalar summaries from a logdir of `events.out.tfevents.*` files via
`tbparse`, which wraps TensorFlow's/TensorBoard's own `EventAccumulator` --
trainspotter never parses the protobuf event format itself.

A logdir is a tree, and what its directories mean depends on what wrote it:

- **One run, one directory** (`transformers.Trainer`, most PyTorch loops):
  tags are already whole names, `train/loss`, `eval/loss`.
- **One run, a directory per split** (Keras): `train/` and `validation/`
  each hold the same tags, `epoch_loss`, `epoch_learning_rate`. The split is
  the directory, so it is put back in front of the tag.
- **One run, a directory per curve** (`SummaryWriter.add_scalars("Loss",
  {"train": a, "val": b})`): `Loss_train/` and `Loss_val/` each hold the tag
  `Loss`. The directory name ends with the curve's own name, which is put
  back behind the tag.
- **Several runs** (`runs/lr3e-4/`, `runs/lr1e-3/`, what `tensorboard
  --logdir runs` is pointed at): one of them is analysed and the others are
  listed (see `select_run`). Merged, two runs are one curve that jumps
  between them at every step.

A run that was resumed from a checkpoint leaves a second event file whose
steps start below where the first one ended. TensorBoard itself drops the
first file's points from that step on when it sees the step counter go
back, and so does this reader; kept, the overlap is two values per step.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath

from trainspotter.model import Run

from .common import build_key_map, select_run

# Directory names that are a split of one run, not a run of their own.
_SPLIT_DIRS = {"train", "training", "validation", "val", "valid", "eval", "test"}


def _place(dir_name: str, tag: str) -> tuple[str, str]:
    """(run, metric name) for a tag found in `dir_name`, relative to the logdir."""
    parts = PurePosixPath(dir_name.replace("\\", "/")).parts
    if parts:
        last = parts[-1]
        if last.lower() in _SPLIT_DIRS:
            return "/".join(parts[:-1]), f"{last}/{tag}"
        # add_scalars: the directory is "<tag with / as _>_<curve>".
        prefix = tag.replace("/", "_") + "_"
        if last.startswith(prefix) and len(last) > len(prefix):
            return "/".join(parts[:-1]), f"{tag}/{last[len(prefix):]}"
    return "/".join(parts), tag


def _drop_overwritten(points: list[tuple[float, int, float]]) -> list[tuple[float, int, float]]:
    """Points as (wall_time, step, value) in the order they were written. When the step counter
    goes back, the run was restarted from there: what was written from that step on is gone."""
    kept: list[tuple[float, int, float]] = []
    for point in points:
        while kept and kept[-1][1] >= point[1]:
            kept.pop()
        kept.append(point)
    return kept


def read_tensorboard(path: str | Path, run_name: str | None = None) -> Run:
    try:
        from tbparse import SummaryReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading TensorBoard event files needs the optional 'tensorboard' extra: "
            "pip install 'trainspotter[tensorboard]'"
        ) from exc

    p = Path(path)
    reader = SummaryReader(str(p), pivot=False, extra_columns={"dir_name", "wall_time"})
    df = reader.scalars
    run = Run(source_format="tensorboard", source_path=str(p))
    if df is None or len(df) == 0:
        return run

    # run -> metric -> points, in file order (tbparse reads a directory's files in name order,
    # which is the order they were created in).
    by_run: dict[str, dict[str, list[tuple[float, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    has_dir = "dir_name" in df.columns
    has_wall_time = "wall_time" in df.columns
    for row in df.itertuples(index=False):
        name, metric = _place(str(row.dir_name) if has_dir else "", str(row.tag))
        wall_time = float(row.wall_time) if has_wall_time else 0.0
        by_run[name][metric].append((wall_time, int(row.step), float(row.value)))

    counts = {name: sum(len(pts) for pts in series.values()) for name, series in by_run.items()}
    chosen = select_run(run, counts, run_name, what=f"{p}: TensorBoard logdir")
    metrics = by_run[chosen]
    names = build_key_map(metrics)
    overwritten = 0
    for metric, points in metrics.items():
        points.sort(key=lambda point: point[0])  # stable: equal wall times keep file order
        kept = _drop_overwritten(points)
        overwritten += len(points) - len(kept)
        for wall_time, step, value in kept:
            run.add_point(names[metric], step, value, wall_time if has_wall_time else None)
    if overwritten:
        run.meta["overwritten_points"] = overwritten
    return run
