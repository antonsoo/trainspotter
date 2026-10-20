"""Read local W&B version-0 transaction logs without a service or API request.

W&B uses its own seven-byte header followed by 32 KiB record blocks. Its
checksums are IEEE CRC32(type + payload), unlike LevelDB's masked CRC32C.
The optional SDK supplies the persistent protobuf schema, not file I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from trainspotter.model import Run

from .common import (
    metric_names,
    metric_value,
    try_step,
    try_typed_float,
    unique_object,
    unreadable_row,
)
from .wandb import _TRAINING_AXES

_HEADER = struct.pack("<4sHB", b":W&B", 0xBEE1, 0)
_BLOCK = 32768
_RECORD_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class _Envelope:
    start: int
    end: int
    fragments: int
    payload: bytes


class _DamagedLogError(ValueError):
    def __init__(self, offset: int, message: str):
        self.offset = offset
        super().__init__(f"byte {offset}: {message}")


def _envelopes(path: Path, limit: int) -> Iterator[_Envelope]:
    """Yield only complete, checksum-valid logical records, in physical order."""
    with path.open("rb") as handle:
        if handle.read(7) != _HEADER:
            raise ValueError("expected a native W&B version-0 transaction-log header")
        pieces: list[bytes] = []
        size = 0
        start = 7
        while handle.tell() < limit:
            position = handle.tell()
            space = _BLOCK - position % _BLOCK
            if space < 7:
                padding = handle.read(min(space, limit - position))
                if len(padding) != space or any(padding):
                    raise _DamagedLogError(
                        start if pieces else position, "incomplete or nonzero padding"
                    )
                continue
            if limit - position < 7:
                raise _DamagedLogError(start if pieces else position, "incomplete record header")
            header = handle.read(7)
            if len(header) != 7:
                raise _DamagedLogError(
                    start if pieces else position, "file changed during a record header"
                )
            checksum, length, kind = struct.unpack("<IHB", header)
            if kind not in (1, 2, 3, 4):
                raise _DamagedLogError(start if pieces else position, "unknown fragment type")
            if length > space - 7:
                raise _DamagedLogError(
                    start if pieces else position, "fragment crosses a block boundary"
                )
            if length > limit - handle.tell():
                raise _DamagedLogError(start if pieces else position, "incomplete fragment payload")
            data = handle.read(length)
            if len(data) != length or zlib.crc32(bytes([kind]) + data) & 0xFFFFFFFF != checksum:
                raise _DamagedLogError(start if pieces else position, "invalid fragment checksum")
            if kind in (1, 2):
                if pieces:
                    raise _DamagedLogError(start, "fragmented record has no final fragment")
                start, size = position, 0
            elif not pieces:
                raise _DamagedLogError(position, "continuation without a first fragment")
            pieces.append(data)
            size += length
            if size > _RECORD_LIMIT:
                raise ValueError("native W&B logical record exceeds the 64 MiB limit")
            if kind in (1, 4):
                yield _Envelope(start, handle.tell(), len(pieces), b"".join(pieces))
                pieces, size = [], 0
        if pieces:
            raise _DamagedLogError(start, "incomplete fragmented record")


def _history(record: Any, paths: dict[str, tuple[str, ...]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for item in record.history.item:
        if item.key and item.nested_key:
            raise ValueError("ambiguous native W&B history key: key and nested_key are both set")
        segments = (item.key,) if item.key else tuple(item.nested_key)
        if not segments or any(not part for part in segments):
            raise ValueError("native W&B history item has no complete key")
        name = ".".join(segments)
        if name in paths and paths[name] != segments:
            raise ValueError(f"native W&B nested and literal history keys collide: {name!r}")
        paths[name] = segments
        if name in values:
            raise ValueError(f"duplicate native W&B history item: {name!r}")
        try:
            # Native JSON can contain observed NaN/Infinity. Keep those as
            # readings; reject malformed or duplicate-member JSON evidence.
            values[name] = json.loads(
                item.value_json,
                object_pairs_hook=unique_object,
                parse_float=Decimal,
            )
        except (ArithmeticError, ValueError, RecursionError) as exc:
            raise ValueError(f"invalid native W&B history JSON for {name!r}: {exc}") from exc
    if record.history.HasField("step"):
        counter = record.history.step.num
        if "_step" in values:
            saved = values["_step"]
            if (
                isinstance(saved, bool)
                or not isinstance(saved, int | float | Decimal)
                or saved != counter
            ):
                raise ValueError("native W&B history _step disagrees with HistoryRecord.step.num")
        values.setdefault("_step", counter)
    return values


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_wandb_native(path: str | Path, *, step_key: str | None = None) -> Run:
    """Decode one native run; a damaged tail cannot become a complete-input pass.

    Two bounded scans establish all metric identities before normalization.
    Repeated steps remain observations; no restart or clock continuity is guessed.
    """
    try:
        from google.protobuf.json_format import MessageToDict
        from google.protobuf.message import DecodeError

        from ._wandb_schema import persistent_record_type

        record_type = persistent_record_type()
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Native .wandb logs need the optional protobuf runtime: "
            "pip install 'protobuf>=5.28.3', or install this checkout with '.[wandb]'"
        ) from exc

    source = Path(path)
    stamp = _stamp(source)
    limit = stamp[2]
    paths: dict[str, tuple[str, ...]] = {}
    raw_names: dict[str, None] = {}
    counts: Counter[str] = Counter()
    run_ids: set[str] = set()
    labels: list[str] = []
    producers: list[str] = []
    headers: list[dict[str, object]] = []
    definitions: list[dict[str, object]] = []
    exits: list[int] = []
    history_records = fragments = 0
    valid_end = 7
    tail_error: str | None = None
    try:
        for envelope in _envelopes(source, limit):
            try:
                record = record_type.FromString(envelope.payload)
            except DecodeError as exc:
                raise _DamagedLogError(envelope.start, "invalid record protobuf") from exc
            original = record.SerializeToString(deterministic=True)
            record.DiscardUnknownFields()
            if record.SerializeToString(deterministic=True) != original:
                raise ValueError(
                    "native W&B record contains fields outside the pinned 0.31.0 schema; "
                    "this reader cannot certify its complete history"
                )
            kind = record.WhichOneof("record_type")
            if kind is None:
                raise ValueError(
                    f"unsupported native W&B persistent record at byte {envelope.start}"
                )
            if kind == "history":
                values = _history(record, paths)
                raw_names.update(dict.fromkeys(values))
                history_records += 1
            elif kind == "run":
                if record.run.run_id:
                    run_ids.add(record.run.run_id)
                if len(run_ids) > 1:
                    raise ValueError("native W&B file contains more than one run identity")
                if record.run.display_name and record.run.display_name not in labels:
                    labels.append(record.run.display_name)
            elif kind == "header":
                headers.append(MessageToDict(record.header, preserving_proto_field_name=True))
                version = record.header.version_info.producer
                if version and version not in producers:
                    producers.append(version)
            elif kind == "exit":
                exits.append(record.exit.exit_code)
            elif kind == "metric":
                definitions.append(
                    {
                        "record_num": record.num,
                        "definition": MessageToDict(
                            record.metric, preserving_proto_field_name=True
                        ),
                    }
                )
            counts[kind] += 1
            fragments += envelope.fragments
            valid_end = envelope.end
    except _DamagedLogError as exc:
        tail_error = str(exc)

    if step_key is None:
        training = [name for name in raw_names if name.strip().lower() in _TRAINING_AXES | {"step"}]
        if len(training) > 1:
            raise ValueError(
                f"multiple native W&B training-step columns: {', '.join(training)}; "
                "choose one with --step-key"
            )
        axis = training[0] if training else "_step"
        if not training and "_step" not in raw_names:
            raise ValueError("native W&B history has no saved logging or training-step counter")
    else:
        if step_key not in raw_names:
            raise ValueError(f"native W&B history has no exact step key {step_key!r}")
        axis = step_key
    run = Run(source_format="wandb-native", source_path=str(source))
    run.meta["step_column"] = axis
    if step_key is not None:
        run.meta["step_selection"] = "explicit"
    elif axis == "_step":
        run.meta["step_axis"] = "logging_step"
    names = metric_names(
        run, (name for name in raw_names if name != axis and not name.startswith("_"))
    )
    skipped = 0
    for envelope in _envelopes(source, valid_end):
        record = record_type.FromString(envelope.payload)
        if record.WhichOneof("record_type") != "history":
            continue
        values = _history(record, paths)
        coordinate = values.get(axis)
        step = (
            try_step(coordinate)
            if isinstance(coordinate, int | float | Decimal) and not isinstance(coordinate, bool)
            else None
        )
        if step is None:
            skipped += 1
            continue
        clock = try_typed_float(values.get("_runtime"))
        wall_time = clock if clock is not None and math.isfinite(clock) else None
        for raw, name in names.items():
            if raw not in values:
                continue
            value = values[raw]
            if value is None:
                # Explicit null is recorded missing evidence, unlike an absent
                # key in a sparse history row. Custom text-only keys stay metadata.
                unreadable_row(run, [name], step, wall_time=wall_time)
                continue
            scalar = metric_value(
                run,
                name,
                value,
                step=step,
                wall_time=wall_time,
                strict_numeric=True,
            )
            if scalar is not None:
                run.add_point(name, step, scalar, wall_time=wall_time)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while data := handle.read(1024 * 1024):
            digest.update(data)
    if _stamp(source) != stamp:
        raise ValueError("native W&B file changed while being read; retry a stable snapshot")
    run.meta["skipped_events"] = skipped
    run.meta["unread_event_bytes"] = limit - valid_end if tail_error is not None else 0
    run.meta["wandb_native"] = {
        "framing_version": 0,
        "file_bytes": limit,
        "sha256": digest.hexdigest(),
        "run_id": next(iter(run_ids)) if run_ids else None,
        "display_names": labels,
        "producer_versions": producers,
        "headers": headers,
        "metric_definitions": definitions,
        "record_types": dict(sorted(counts.items())),
        "records_read": sum(counts.values()),
        "fragments_read": fragments,
        "history_records": history_records,
        "exit_codes": exits,
        "unread_from_offset": valid_end if tail_error is not None else None,
        "tail_error": tail_error,
        "history_key_paths": {name: list(parts) for name, parts in paths.items()},
    }
    return run
