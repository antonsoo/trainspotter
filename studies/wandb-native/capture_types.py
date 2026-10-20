"""SDK-native numeric-looking text; it must not become a numeric observation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_ERROR_REPORTING"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import wandb  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    settings = wandb.Settings(
        x_disable_stats=True,
        x_disable_meta=True,
        x_disable_machine_info=True,
        disable_git=True,
        disable_code=True,
        disable_job_creation=True,
        console="off",
        host="synthetic-cpu",
        username="native-study",
    )
    run = wandb.init(
        mode="offline",
        dir=out,
        project="trainspotter-offline-study",
        id="native-typed-values",
        name="Typed scalar evidence control",
        settings=settings,
    )
    values = [1.0, "0.5", "NaN", 0.1]
    for step, value in enumerate(values):
        run.log({"global_step": step, "train/loss": value})
    path = Path(run.settings.sync_file)
    run.finish()
    # A short transaction log may remain buffered after the run acknowledgement.
    # Closing the SDK service seals the file before this snapshot is copied.
    wandb.teardown()
    shutil.copyfile(path, out / "run.wandb")
    data = (out / "run.wandb").read_bytes()
    assert len(data) > 7
    assert not any(token in data for token in (b"/home/", b"/tmp/", b"/mnt/", b"antonsoloviev"))
    reference = {
        "scope": "authored SDK-native type control, not a training run",
        "logged_values": values,
        "native_json_types": ["number", "string", "string", "number"],
        "expected_numeric_steps": [0, 3],
        "expected_unreadable_steps": [1, 2],
        "sdk_version": importlib.metadata.version("wandb"),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "service_closed_before_snapshot": True,
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    main()
