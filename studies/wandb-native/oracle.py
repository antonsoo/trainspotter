"""Decode unchanged captures with two official, independent W&B file readers.

Use W&B 0.23.1 for python-sdk (its DataStore), and W&B 0.31.0 for leet (Go).
Neither mode imports trainspotter. The old SDK opens its input r+b, so it reads
a temporary byte-identical copy; the retained capture is never opened writable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def normalized(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and value in ("NaN", "Infinity", "-Infinity"):
        return str(float(value))
    return value


def sdk(path: Path) -> dict[str, Any]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    kinds: Counter[str] = Counter()
    fragments = 0
    rows = []
    long_records = []
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "copy.wandb"
        shutil.copyfile(path, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        original_scan = store.scan_record

        def count_fragment() -> object:
            nonlocal fragments
            result = original_scan()
            if result is not None:
                fragments += 1
            return result

        store.scan_record = count_fragment
        while True:
            start = store.get_offset()
            data = store.scan_data()
            if data is None:
                break
            end = store.get_offset()
            record = wandb_internal_pb2.Record.FromString(data)
            kind = record.WhichOneof("record_type")
            assert kind is not None
            kinds[kind] += 1
            if len(data) > 32761:
                long_records.append(
                    {
                        "record_num": record.num,
                        "kind": kind,
                        "start": start,
                        "end": end,
                        "payload_bytes": len(data),
                    }
                )
            if kind != "history":
                continue
            values = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                assert key not in values
                if key == "logger.note":
                    text = json.loads(item.value_json)
                    assert len(text) == 100_017
                    continue
                values[key] = normalized(json.loads(item.value_json))
            assert int(values["_step"]) == record.history.step.num
            rows.append({"record_num": record.num, "start": start, "end": end, "items": values})
        store.close()
    module = Path(inspect.getsourcefile(DataStore))
    return {
        "reader": "official Python DataStore",
        "record_types": dict(sorted(kinds.items())),
        "fragments_read": fragments,
        "history": rows,
        "long_records": long_records,
        "decoder_source_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }


def leet(path: Path) -> dict[str, Any]:
    from wandb.util import get_core_path

    core = Path(get_core_path())
    result = subprocess.run(
        [
            str(core),
            "leet",
            "inspect",
            "--no-observability",
            "--json",
            "--run-file",
            str(path.resolve()),
            str(path.resolve().parent.parent),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    kinds = Counter(record["type"] for record in records)
    rows = []
    for record in records:
        if record["type"] != "history":
            continue
        values = record["history"]["item"]
        note = values.pop("logger.note", None)
        if note is not None:
            assert len(note) == 100_017
        rows.append(
            {
                "record_num": record["num"],
                "items": {key: normalized(value) for key, value in values.items()},
            }
        )
    return {
        "reader": "official Go LEET inspect --json",
        "record_types": dict(sorted(kinds.items())),
        "history": rows,
        "decoder_binary_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "stderr": result.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("python-sdk", "leet"))
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "producer.json").read_text())
    cases = []
    for item in manifest["cases"]:
        path = args.capture / item["file"]
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        assert original == item["sha256"]
        result = sdk(path) if args.mode == "python-sdk" else leet(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original
        assert len(result["history"]) == item["logging_calls"]
        cases.append({"case": item["case"], "file_sha256": original, **result})
    receipt = {
        "mode": args.mode,
        "sdk_version": importlib.metadata.version("wandb"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_manifest_sha256": hashlib.sha256(
            (args.capture / "producer.json").read_bytes()
        ).hexdigest(),
        "cases": cases,
    }
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(args.mode, "decoded", sum(len(case["history"]) for case in cases), "native histories")


if __name__ == "__main__":
    main()
