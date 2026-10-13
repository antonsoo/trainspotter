"""Exercise actual CLI report destinations using a retained TensorBoard SDK capture.

The output directory must be new: every attempted overwrite targets disposable
copies. No training run or pre-existing user report is modified by this replay.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expect", choices=["before", "after"], default="after")
    args = parser.parse_args()
    python = str(args.python.absolute())  # Do not resolve a venv's executable symlink.
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve())
    original = Path(__file__).with_name("events.out.tfevents.controlled").read_bytes()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    receipts = []
    transcript = []

    def invoke(directory, output, destination=None):
        command = [python, "-m", "trainspotter.cli", "analyze", "logs", "--run", "a",
                   "--output", output]
        if destination is not None:
            command += ["--out", str(destination)]
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                text=True, timeout=30)
        return result

    for mode in ["terminal", "json", "html"]:
        for alias in ["selected", "unselected", "symlink", "hardlink", "parent-symlink",
                      "new-event", "broken-event-link"]:
            label = f"{mode}-{alias}"
            directory = artifacts / label
            for name in ["a", "b"]:
                writer = directory / "logs" / name / "train"
                writer.mkdir(parents=True)
                (writer / "events.out.tfevents.controlled").write_bytes(original)
            selected = Path("logs/a/train/events.out.tfevents.controlled")
            other = Path("logs/b/train/events.out.tfevents.controlled")
            target = selected if alias == "selected" else other
            if alias in ["symlink", "hardlink"]:
                target = Path("report-alias")
                if alias == "symlink":
                    (directory / target).symlink_to(other)
                else:
                    (directory / target).hardlink_to(directory / other)
            elif alias == "parent-symlink":
                (directory / "writer-alias").symlink_to("logs/b/train", target_is_directory=True)
                target = Path("writer-alias/events.out.tfevents.controlled")
            elif alias in ["new-event", "broken-event-link"]:
                target = Path("logs/report.tfevents.json")
                if alias == "broken-event-link":
                    (directory / target).symlink_to("missing")
            existed = (directory / target).exists()
            was_link = (directory / target).is_symlink()
            previous = (directory / target).read_bytes() if existed else None
            result = invoke(directory, mode, target)
            native_unchanged = all((directory / file).read_bytes() == original
                                   for file in [selected, other])
            target_unchanged = ((directory / target).read_bytes() == previous if existed
                                else not (directory / target).exists())
            link_unchanged = (directory / target).is_symlink() == was_link
            assert result.returncode == (0 if args.expect == "before" else 2), result.stderr
            if args.expect == "after":
                assert native_unchanged and target_unchanged and link_unchanged, label
                assert not list(directory.rglob(".trainspotter-*")), label
            else:
                assert not target_unchanged or not link_unchanged, label
            receipts.append({"case": label, "exit": result.returncode,
                             "nativeInputsUnchanged": native_unchanged,
                             "destinationUnchanged": target_unchanged,
                             "linkUnchanged": link_unchanged, "stderr": result.stderr.strip()})
            if mode == "json" and alias in ["selected", "unselected", "new-event"]:
                transcript += [f"$ trainspotter analyze logs --run a --output json --out {target}",
                               result.stderr.strip(), f"exit={result.returncode}; native files unchanged={native_unchanged}", ""]

    # Ordinary reports inside the tree still work, and a second read remains complete.
    directory = artifacts / "safe"
    writer = directory / "logs/a/train"
    writer.mkdir(parents=True)
    event = writer / "events.out.tfevents.controlled"
    event.write_bytes(original)
    safe = []
    for mode, suffix in [("terminal", "txt"), ("json", "json"), ("html", "html")]:
        target = Path("logs") / f"report.{suffix}"
        (directory / target).write_text("previous report", encoding="utf-8")
        result = invoke(directory, mode, target)
        assert result.returncode == 0, result.stderr
        assert event.read_bytes() == original
        data = (directory / target).read_bytes()
        assert data != b"previous report"
        safe.append({"output": mode, "bytes": len(data), "sha256": digest(data)})
    reread = invoke(directory, "json")
    assert reread.returncode == 0, reread.stderr
    report = json.loads(reread.stdout)
    assert report["run"]["input_complete"]
    assert report["run"]["metrics"]["train/loss"]["count"] == 20
    info = subprocess.run([python, "-c", "import json,sys,tensorboard,trainspotter; "
                           "print(json.dumps({'python':sys.version.split()[0], 'tensorboard':tensorboard.__version__, 'trainspotter':trainspotter.__version__}))"],
                          cwd=artifacts, env=env, capture_output=True, text=True, check=True)
    result = {"expectation": args.expect, "environment": json.loads(info.stdout),
              "fixture": {"bytes": len(original), "sha256": digest(original),
                          "provenance": "TensorBoard 2.21.0 EventFileWriter; 20 synthetic scalar loss readings. No model or private data."},
              "guardCases": receipts, "safeReports": safe,
              "safeReread": {"inputComplete": True, "lossReadings": 20}}
    (artifacts / "replay.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (artifacts / "terminal.txt").write_text("\n".join(transcript), encoding="utf-8")
    print(json.dumps({"guardCases": len(receipts), "safeReports": len(safe),
                      "inputCompleteAfterWritingReports": True}))


if __name__ == "__main__":
    main()
