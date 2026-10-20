"""Extract only persistent descriptors from the independently installed W&B SDK."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from google.protobuf import descriptor_pb2
from wandb.proto import wandb_internal_pb2


def main() -> None:
    assert importlib.metadata.version("wandb") == "0.31.0"
    destination = Path(sys.argv[1])
    descriptors = descriptor_pb2.FileDescriptorSet()
    seen = set()

    def add(source):
        if source.name in seen:
            return
        for child in source.dependencies:
            add(child)
        seen.add(source.name)
        descriptors.file.add().ParseFromString(source.serialized_pb)

    add(wandb_internal_pb2.Record.DESCRIPTOR.file)
    data = descriptors.SerializeToString(deterministic=True)
    assert len(data) == 30_346 and len(seen) == 6
    destination.write_bytes(data)
    receipt = {
        "sdk": "wandb==0.31.0",
        "protobuf": importlib.metadata.version("protobuf"),
        "root_message": "wandb_internal.Record",
        "scope": "Unmodified persistent FileDescriptorProto messages in dependency order.",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "descriptors": [
            {
                "name": file.name,
                "sha256": hashlib.sha256(file.SerializeToString(deterministic=True)).hexdigest(),
            }
            for file in descriptors.file
        ],
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Extracted", len(data), "bytes of unmodified persistent descriptors")


if __name__ == "__main__":
    main()
