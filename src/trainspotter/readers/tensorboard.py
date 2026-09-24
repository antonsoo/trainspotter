"""TensorBoard event-file reader (optional extra: `pip install trainspotter[tensorboard]`).

Reads scalar summaries from a logdir of `events.out.tfevents.*` files via
`tbparse`, which wraps TensorFlow's/TensorBoard's own `EventAccumulator` --
trainspotter never parses the protobuf event format itself.
"""

from __future__ import annotations

from pathlib import Path

from trainspotter.model import Run

from .common import normalize_key


def read_tensorboard(path: str | Path) -> Run:
    try:
        from tbparse import SummaryReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading TensorBoard event files needs the optional 'tensorboard' extra: "
            "pip install 'trainspotter[tensorboard]'"
        ) from exc

    p = Path(path)
    reader = SummaryReader(str(p), pivot=False)
    df = reader.scalars
    run = Run(source_format="tensorboard", source_path=str(p))
    if df is None or len(df) == 0:
        return run
    has_wall_time = "wall_time" in df.columns
    for row in df.itertuples(index=False):
        tag = str(row.tag)
        step = int(row.step)
        value = float(row.value)
        wall_time = float(row.wall_time) if has_wall_time else None
        run.add_point(normalize_key(tag), step, value, wall_time=wall_time)
    return run
