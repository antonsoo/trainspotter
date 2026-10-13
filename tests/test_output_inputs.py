"""Actual SDK writer files must survive every CLI output mode and path alias."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainspotter import cli
from trainspotter.readers import load_run
from trainspotter.readers.tensorboard import event_files

writer = pytest.importorskip("tensorboard.summary.writer.event_file_writer")
event_pb2 = pytest.importorskip("tensorboard.compat.proto.event_pb2")
summary_pb2 = pytest.importorskip("tensorboard.compat.proto.summary_pb2")


def capture(directory: Path) -> Path:
    stream = writer.EventFileWriter(str(directory))
    for step in range(20):
        stream.add_event(event_pb2.Event(
            wall_time=1700000000 + step, step=step,
            summary=summary_pb2.Summary(value=[
                summary_pb2.Summary.Value(tag="loss", simple_value=1 / (step + 1)),
            ]),
        ))
    stream.close()
    return next(directory.glob("*tfevents.*"))


@pytest.mark.parametrize("output", ["terminal", "json", "html"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "parent-symlink"])
def test_directory_input_survives_report_aliases(tmp_path, output, alias):
    root = tmp_path / "logs"
    selected = capture(root / "a" / "train")
    other = capture(root / "b" / "train")
    before = {file: file.read_bytes() for file in [selected, other]}
    # Selecting a must not make b's training data available as report scratch space.
    target = other
    if alias == "symlink":
        target = tmp_path / "report-link"
        target.symlink_to(other)
    elif alias == "hardlink":
        target = tmp_path / "report-hardlink"
        target.hardlink_to(other)
    elif alias == "parent-symlink":
        parent = tmp_path / "writer-alias"
        parent.symlink_to(other.parent, target_is_directory=True)
        target = parent / other.name
    inode = target.stat().st_ino
    assert cli.main([
        "analyze", str(root), "--run", "a", "--output", output, "--out", str(target),
    ]) == 2
    assert target.stat().st_ino == inode
    assert all(file.read_bytes() == data for file, data in before.items())
    assert len(load_run(root, run_name="b").get("train/loss").points) == 20
    assert not list(target.parent.glob(".trainspotter-*"))


def test_selected_event_and_symlinked_source_directory_are_protected(tmp_path):
    root = tmp_path / "logs"
    event = capture(root / "train")
    alias = tmp_path / "source-alias"
    alias.symlink_to(root, target_is_directory=True)
    before = event.read_bytes()
    assert cli.main(["analyze", str(alias), "--out", str(event)]) == 2
    assert event.read_bytes() == before


@pytest.mark.parametrize("alias", [False, True])
def test_new_report_cannot_become_an_event_on_the_next_read(tmp_path, alias):
    root = tmp_path / "logs"
    event = capture(root / "train")
    parent = root / "train"
    if alias:
        parent = tmp_path / "parent-alias"
        parent.symlink_to(root / "train", target_is_directory=True)
    target = parent / "report.tfevents.json"
    before = event.read_bytes()
    assert cli.main(["analyze", str(root), "--output", "json", "--out", str(target)]) == 2
    assert not target.exists()
    assert event.read_bytes() == before


def test_replacing_event_named_link_would_poison_the_directory(tmp_path):
    root = tmp_path / "logs"
    capture(root / "train")
    # Broken links are not existing reader files; replacement would create one.
    target = root / "report.tfevents.json"
    target.symlink_to(tmp_path / "missing")
    assert cli.main(["analyze", str(root), "--output", "json", "--out", str(target)]) == 2
    assert target.is_symlink()
    assert not target.exists()


def test_safe_report_replacement_in_logdir_remains_atomic_and_readable(tmp_path):
    root = tmp_path / "logs"
    event = capture(root / "train")
    ignored = root / "events.tfevents.ignored.profile-empty"
    ignored.write_bytes(b"profile placeholder")
    target = root / "report.json"
    target.write_text("old report")
    original = event.read_bytes()
    assert event_files(root) == [event]
    assert cli.main(["analyze", str(root), "--output", "json", "--out", str(target)]) == 0
    assert json.loads(target.read_text())["run"]["input_complete"]
    assert event.read_bytes() == original
    assert len(load_run(root).get("train/loss").points) == 20
    assert not list(root.glob(".trainspotter-*"))
    # Outside the watched directory, the event-like name cannot contaminate input.
    outside = tmp_path / "report.tfevents.json"
    assert cli.main(["analyze", str(root), "--output", "json", "--out", str(outside)]) == 0
    assert json.loads(outside.read_text())["run"]["input_complete"]
