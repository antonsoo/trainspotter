"""An official-SDK zero-length FIRST fragment at a 32 KiB block boundary.

Run with W&B 0.23.1, which still includes its Python DataStore writer. This is
an authored protocol boundary control, not a captured training experiment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


def main() -> None:
    path = Path(sys.argv[1])
    padding = pb.Record(num=1)
    item = padding.config.update.add(key="padding")
    count = 32_700
    while True:
        item.value_json = json.dumps("x" * count)
        size = len(padding.SerializeToString())
        if size == 32_747:
            break
        count += 32_747 - size
    history = pb.Record(num=2)
    history.history.step.num = 0
    history.history.item.add(key="train/loss", value_json="1")
    store = DataStore()
    store.open_for_write(str(path))
    store.write(padding)
    assert store.get_offset() == 32761
    store.write(history)
    store.close()
    data = path.read_bytes()
    checksum, size, kind = struct.unpack("<IHB", data[32761:32768])
    assert size == 0 and kind == 2
    receipt = {
        "purpose": "SDK-written boundary control; seven bytes left -> empty FIRST, then LAST",
        "wandb": importlib.metadata.version("wandb"),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "empty_first_offset": 32761,
        "empty_first_checksum": checksum,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
