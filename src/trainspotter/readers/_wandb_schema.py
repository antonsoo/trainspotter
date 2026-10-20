"""Load the pinned persistent schema without importing or installing a W&B client."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def persistent_record_type() -> Any:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    schema = descriptor_pb2.FileDescriptorSet.FromString(
        files("trainspotter.readers").joinpath("wandb-schema.pb").read_bytes()
    )
    # A private pool keeps the pinned schema separate from protobuf definitions
    # a caller may have loaded through another SDK version in this process.
    pool = descriptor_pool.DescriptorPool()
    for descriptor in schema.file:
        pool.AddSerializedFile(descriptor.SerializeToString())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("wandb_internal.Record"))
