"""Unsampled TensorBoard scalars in event order, including tensor-valued scalars.

TensorBoard decodes the TFRecords and protobufs. We retain scalar readings only,
following EventAccumulator's restart rules: v2 uses SessionLog.START to invalidate
all tags in that writer directory at or beyond its step; older event streams use
the per-tag step-rewind heuristic. Wall clocks never decide which readings survive.
See docs/input-integrity.md for the contract and independent oracle tests.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from trainspotter.model import Point, Run

from .common import metric_names, select_run, try_step

_SPLIT_DIRS = {"train", "training", "validation", "val", "valid", "eval", "test"}


def _place(dir_name: str, tag: str) -> tuple[str, str]:
    """(run, metric name) for a tag found relative to the requested logdir."""
    parts = PurePosixPath(dir_name.replace("\\", "/")).parts
    if parts:
        last = parts[-1]
        if last.lower() in _SPLIT_DIRS:
            return "/".join(parts[:-1]), f"{last}/{tag}"
        prefix = tag.replace("/", "_") + "_"
        if last.startswith(prefix) and len(last) > len(prefix):
            return "/".join(parts[:-1]), f"{tag}/{last[len(prefix):]}"
    return "/".join(parts), tag


@dataclass
class _Directory:
    points: dict[str, list[Point]] = field(default_factory=lambda: defaultdict(list))
    scalar_tags: set[str] = field(default_factory=set)
    plugins: dict[str, str] = field(default_factory=dict)
    version: float | None = None
    most_recent_step: int = -1
    overwritten: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    restarts: int = 0
    skipped: int = 0
    unread_bytes: int = 0

    def purge(self, step: int, tags: list[str]) -> None:
        for tag in dict.fromkeys(tags):
            old = self.points.get(tag, [])
            kept = [point for point in old if point.step < step]
            self.overwritten[tag] += sum(p.value is not None and p.step >= step for p in old)
            if tag in self.points:
                self.points[tag] = kept

    def consume(self, event: Any) -> None:
        from tensorboard.backend.event_processing.event_util import ParseFileVersion
        from tensorboard.compat.proto.event_pb2 import SessionLog
        from tensorboard.util.tensor_util import make_ndarray

        if event.HasField("file_version"):
            self.version = ParseFileVersion(event.file_version)
        step = try_step(event.step)
        if step is None:
            if event.HasField("summary") or event.HasField("session_log"):
                self.skipped += 1
                self.scalar_tags.update(
                    value.tag or value.node_name for value in event.summary.value
                )
            return
        if self.version is not None and self.version >= 2:
            if event.HasField("session_log") and event.session_log.status == SessionLog.START:
                self.purge(step, list(self.points))
                self.restarts += 1
        elif step < self.most_recent_step and event.HasField("summary"):
            self.purge(step, [value.tag for value in event.summary.value])
        else:
            self.most_recent_step = step

        wall_time = event.wall_time if math.isfinite(event.wall_time) else None
        for value in event.summary.value:
            tag = value.tag or value.node_name
            if value.HasField("metadata"):
                self.plugins.setdefault(tag, value.metadata.plugin_data.plugin_name)
            if value.HasField("simple_value"):
                self.scalar_tags.add(tag)
                number = float(value.simple_value)
            elif value.HasField("tensor") and self.plugins.get(tag, "") in ("", "scalars"):
                tensor = value.tensor
                # Check shape BEFORE decoding: a vector/image isn't a scalar, and an
                # untrusted huge shape must not trigger an allocation just to discover it.
                if tensor.tensor_shape.unknown_rank or tensor.tensor_shape.dim:
                    if self.plugins.get(tag) == "scalars":
                        self.scalar_tags.add(tag)
                        self.skipped += 1
                        self.points[tag].append(Point(step, None, wall_time))
                    continue
                try:
                    array = make_ndarray(tensor)
                    if array.dtype.kind not in "iuf":
                        if self.plugins.get(tag) == "scalars":
                            self.scalar_tags.add(tag)
                            self.skipped += 1
                            self.points[tag].append(Point(step, None, wall_time))
                        continue
                    number = float(array.item())
                except (ValueError, TypeError, OverflowError, KeyError):
                    self.scalar_tags.add(tag)
                    self.skipped += 1
                    self.points[tag].append(Point(step, None, wall_time))
                    continue
                self.scalar_tags.add(tag)
            else:
                continue
            self.points[tag].append(Point(step, number, wall_time))


def _read_directory(files: list[Path]) -> _Directory:
    from google.protobuf.message import DecodeError
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event

    data = _Directory()
    for path in files:
        read_bytes = 0
        records = iter(RawEventFileLoader(str(path)).Load())
        while True:
            try:
                record = next(records)
            except StopIteration:
                break
            except OverflowError:
                # A damaged uint64 length can exceed Python's read() index range
                # in TensorBoard's standalone decoder. Retain preceding readings
                # and count the whole remaining tail as unread below.
                break
            # TFRecord framing: 8-byte length + 4-byte length CRC + 4-byte data CRC.
            read_bytes += len(record) + 16
            try:
                data.consume(Event.FromString(record))
            except DecodeError:
                data.skipped += 1
        data.unread_bytes += max(0, path.stat().st_size - read_bytes)
    return data


def _possible_runs(directory: str) -> set[str]:
    """Owners an undecodable record could have under this reader's layout rules.

    A split directory always belongs to its parent. An add_scalars name only
    identifies its parent once a matching tag is read; with unread records an
    underscore directory can belong to itself or its parent. Do not turn that
    uncertainty into a clean gate for either candidate.
    """
    parts = PurePosixPath(directory).parts
    if parts and parts[-1].lower() in _SPLIT_DIRS:
        return {"/".join(parts[:-1])}
    owners = {"/".join(parts)}
    if parts and "_" in parts[-1][:-1]:
        owners.add("/".join(parts[:-1]))
    return owners


def read_tensorboard(path: str | Path, run_name: str | None = None) -> Run:
    try:
        import tensorboard  # noqa: F401
    except ImportError as exc:  # pragma: no cover - checked in the package smoke test
        raise ImportError(
            "reading TensorBoard event files needs the optional 'tensorboard' extra: "
            "pip install 'trainspotter[tensorboard]'"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TensorBoard source does not exist: {p}")
    paths = sorted(p.rglob("*tfevents.*")) if p.is_dir() else [p]
    files_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for file in paths:
        if file.is_file() and not file.name.endswith(".profile-empty"):
            files_by_dir[file.parent].append(file)
    root = p if p.is_dir() else p.parent
    by_run: dict[str, list[tuple[str, str, list[Point]]]] = defaultdict(list)
    overwritten: dict[str, int] = defaultdict(int)
    restarts: dict[str, int] = defaultdict(int)
    run = Run(source_format="tensorboard", source_path=str(p))
    directories: list[dict[str, Any]] = []
    for directory, files in files_by_dir.items():
        data = _read_directory(files)
        relative = directory.relative_to(root).as_posix()
        owners = {_place(relative, tag)[0] for tag in data.scalar_tags}
        damaged = bool(data.skipped or data.unread_bytes)
        discovered = owners or ({_place(relative, "")[0]} if damaged else set())
        for name in discovered:
            by_run.setdefault(name, [])  # Damaged or fully purged runs stay selectable.
        for tag, points in data.points.items():
            name, metric = _place(relative, tag)
            overwritten[name] += data.overwritten.get(tag, 0)
            if points:
                by_run[name].append((metric, f"{relative}: {tag}", points))
        for name in owners:
            restarts[name] += data.restarts
        candidates = owners | _possible_runs(relative) if damaged else owners
        directories.append({
            "directory": relative, "observed_runs": sorted(owners),
            "possible_runs": sorted(candidates), "skipped_events": data.skipped,
            "unread_event_bytes": data.unread_bytes,
        })
    run.meta["event_directories"] = directories
    if not by_run and run_name is None:
        return run
    counts = {name: sum(p.value is not None for _, _, points in series for p in points)
              for name, series in by_run.items()}
    chosen = select_run(run, counts, run_name, what=f"{p}: TensorBoard logdir")
    for record in directories:
        record["selected"] = chosen in record["possible_runs"]
    run.meta["skipped_events"] = sum(r["skipped_events"] for r in directories if r["selected"])
    run.meta["unread_event_bytes"] = sum(
        r["unread_event_bytes"] for r in directories if r["selected"]
    )
    run.meta["selected_run"] = chosen
    run.meta["overwritten_points"] = overwritten[chosen]
    run.meta["restart_markers"] = restarts[chosen]
    metrics = by_run[chosen]
    # Distinct writer directories can spell the same logical metric. Preserve their
    # source identities instead of concatenating independent curves.
    occurrences: dict[str, int] = defaultdict(int)
    for metric, _, _ in metrics:
        occurrences[metric] += 1
    run.meta["ambiguous_metric_sources"] = [
        metric for metric, count in occurrences.items() if count > 1
    ]
    raw_names = [
        metric if occurrences[metric] == 1 else f"{metric} [{source}]"
        for metric, source, _ in metrics
    ]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("TensorBoard metric names are ambiguous across writer directories")
    names = metric_names(run, raw_names)
    run.meta["metric_sources"] = {
        names[raw]: source for raw, (_, source, _) in zip(raw_names, metrics, strict=True)
    }
    for raw, (_, _, points) in zip(raw_names, metrics, strict=True):
        for point in points:
            run.add_point(names[raw], point.step, point.value, point.wall_time)
    return run
