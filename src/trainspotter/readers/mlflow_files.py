"""Legacy MLflow FileStore histories. PyYAML is needed only for run metadata."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from trainspotter.model import Run

from .common import metric_names, try_float, try_step
from .mlflow import _issue, _observation, _RunInfo, _select

_MAX_METADATA_BYTES = 1024 * 1024
_MAX_METRIC_LINE = 64 * 1024
_STATUSES = {1: "RUNNING", 2: "SCHEDULED", 3: "FINISHED", 4: "FAILED", 5: "KILLED"}
_META_FIELDS = {"run_id", "run_uuid", "run_name", "experiment_id", "status", "lifecycle_stage"}


def run_directories(root: Path) -> list[Path]:
    """Store, experiment or individual run; never recurse into model artifacts."""
    found = []
    pending = [(root, 0)]
    # FileStore moves deleted experiments (with their run directories) here.
    if (root / ".trash").is_dir():
        pending.append((root / ".trash", 0))
    while pending:
        folder, depth = pending.pop()
        if ((folder / "meta.yaml").is_file()
                and ((folder / "metrics").is_dir() or (folder / "tags/mlflow.runName").is_file())):
            found.append(folder)
        elif depth < 2:
            pending.extend((child, depth + 1) for child in sorted(folder.iterdir())
                           if child.is_dir() and not child.name.startswith("."))
    return sorted(found)


def is_file_store(path: Path) -> bool:
    return path.is_dir() and bool(run_directories(path))


def _metadata(folder: Path) -> _RunInfo:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "MLflow FileStore metadata needs PyYAML: pip install 'PyYAML>=6.0.3'. "
            "The source checkout also offers the mlflow-files extra. "
            "Native SQLite reading needs neither."
        ) from exc
    path = folder / "meta.yaml"
    with path.open("rb") as handle:
        raw = handle.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError(f"{path}: MLflow metadata exceeds 1 MiB")
    data: dict[str, object] = {}
    try:
        loader = yaml.SafeLoader(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: unreadable MLflow run metadata ({exc})") from exc
    try:
        node = loader.get_single_node()
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("run metadata must be a mapping")
        keys: set[str] = set()
        for key_node, value_node in node.value:
            if (not isinstance(key_node, yaml.ScalarNode)
                    or key_node.tag != "tag:yaml.org,2002:str" or key_node.value in keys):
                raise ValueError("run metadata has a non-string or repeated key")
            key = key_node.value
            keys.add(key)
            if key in _META_FIELDS:
                if not isinstance(value_node, yaml.ScalarNode):
                    raise ValueError(f"{key} must be a scalar in run metadata")
                data[key] = loader.construct_object(value_node)
    except (yaml.YAMLError, RecursionError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable MLflow run metadata ({exc})") from exc
    finally:
        loader.dispose()
    key = data.get("run_id", data.get("run_uuid"))
    if (not isinstance(key, str) or not key
            or ("run_uuid" in data and data["run_uuid"] != key)):
        raise ValueError(f"{path}: missing or conflicting MLflow run identity")
    lifecycle = data.get("lifecycle_stage")
    if lifecycle not in ("active", "deleted"):
        raise ValueError(f"{path}: unsupported MLflow lifecycle stage")
    experiment = data.get("experiment_id")
    if not isinstance(experiment, str | int) or isinstance(experiment, bool):
        raise ValueError(f"{path}: missing MLflow experiment identity")
    label = data.get("run_name")
    if label is None and (folder / "tags/mlflow.runName").is_file():
        with (folder / "tags/mlflow.runName").open(encoding="utf-8") as handle:
            label = handle.read(_MAX_METADATA_BYTES + 1)
        if len(label) > _MAX_METADATA_BYTES:
            raise ValueError(f"{path}: MLflow display name exceeds 1 MiB")
    status = data.get("status")
    status_name = _STATUSES.get(status, "unknown") if type(status) is int else str(status)
    return _RunInfo(key, label if isinstance(label, str) else key, str(experiment),
                    status_name, str(lifecycle))


@dataclass(frozen=True)
class _MetricFile:
    key: str
    path: Path
    size: int
    device: int
    inode: int


def _records(source: _MetricFile) -> Iterator[tuple[int, bytes | None, str | None]]:
    """Read the observed byte prefix; an appended row waits until the next poll."""
    with source.path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        if ((stat.st_dev, stat.st_ino) != (source.device, source.inode)
                or stat.st_size < source.size):
            raise ValueError(f"MLflow metric file changed while reading: {source.path}; retry")
        remaining = source.size
        line = 0
        while remaining:
            raw = handle.readline(min(remaining, _MAX_METRIC_LINE + 1))
            if not raw:
                raise ValueError(f"MLflow metric file was truncated while reading: {source.path}")
            remaining -= len(raw)
            line += 1
            oversized = len(raw) > _MAX_METRIC_LINE
            while remaining and not raw.endswith(b"\n"):
                oversized = True
                raw = handle.readline(min(remaining, _MAX_METRIC_LINE + 1))
                if not raw:
                    raise ValueError(
                        f"MLflow metric file was truncated while reading: {source.path}"
                    )
                remaining -= len(raw)
            if oversized:
                yield line, None, "metric record exceeds 64 KiB"
            elif not raw.endswith(b"\n"):
                yield line, None, "unterminated metric record; the writer may still be appending"
            else:
                yield line, raw, None


def _integer(raw: str) -> int | None:
    return try_step(raw) if re.fullmatch(r"[+-]?[0-9]+", raw) else None


def read_file_store(path: Path, wanted: str | None, step_key: str | None) -> Run:
    run = Run(source_format="mlflow", source_path=str(path))
    run.meta["step_column"] = "step"
    run.meta["step_axis"] = "step" if step_key else "logging_step"
    if step_key:
        run.meta["step_selection"] = "explicit"
    infos: dict[str, _RunInfo] = {}
    snapshots: dict[str, list[_MetricFile]] = {}
    counts = {}
    directories = run_directories(path)
    for folder in directories:
        info = _metadata(folder)
        if info.run_id in infos:
            raise ValueError(
                f"MLflow run ID {info.run_id!r} occurs in multiple directories; "
                "select one run directory instead of merging its copies"
            )
        files = []
        for file in sorted((folder / "metrics").rglob("*")):
            if file.is_file():
                stat = file.stat()
                files.append(_MetricFile(file.relative_to(folder / "metrics").as_posix(), file,
                                         stat.st_size, stat.st_dev, stat.st_ino))
        infos[info.run_id] = info
        snapshots[info.run_id] = files
        counts[info.run_id] = sum(1 for file in files for _ in _records(file))
    if len(infos) == 1 and path == directories[0] and wanted is None:
        wanted = next(iter(infos))
    chosen = _select(run, infos, counts, wanted, "files")
    evidence = run.meta["mlflow"]
    assert isinstance(evidence, dict)
    evidence["metric_files"] = {file.key: {"path": str(file.path.relative_to(path)),
                                         "snapshot_bytes": file.size}
                                for file in snapshots[chosen]}
    names = metric_names(run, [file.key for file in snapshots[chosen]])
    for source in snapshots[chosen]:
        metric = names[source.key]
        for line, raw, problem in _records(source):
            if problem:
                _issue(run, metric, f"line {line}: {problem}")
                continue
            assert raw is not None
            try:
                fields = raw.decode("utf-8-sig" if line == 1 else "utf-8").split()
            except UnicodeDecodeError:
                _issue(run, metric, f"line {line}: malformed UTF-8 metric record")
                continue
            if len(fields) not in (2, 3):
                step = _integer(fields[2]) if len(fields) >= 3 else None
                reason = ("dataset-qualified records need separate metric histories"
                          if len(fields) == 5 else "expected timestamp, value and optional step")
                _issue(run, metric, f"line {line}: {reason}", step)
                if step is not None:
                    run.add_unreadable(metric, step, known_metric=True)
                continue
            step = _integer(fields[2]) if len(fields) == 3 else 0
            _observation(run, metric, step, try_float(fields[1]), _integer(fields[0]))
    return run


def output_conflict(source: Path, destination: Path) -> bool:
    """Keep reports outside live metric, tag and parameter files; artifacts are allowed."""
    target = destination.resolve()
    protected = [source / "meta.yaml"]
    protected.extend(folder / "meta.yaml" for folder in source.iterdir() if folder.is_dir())
    for folder in run_directories(source):
        protected.append(folder / "meta.yaml")
        protected.append(folder.parent / "meta.yaml")
        for name in ("metrics", "params", "tags"):
            directory = (folder / name).resolve()
            if target.is_relative_to(directory):
                return True
            protected.extend(file for file in (folder / name).rglob("*") if file.is_file())
    return any(target == file.resolve() or (
        destination.exists() and file.exists() and destination.samefile(file)
    ) for file in protected)
