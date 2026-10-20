from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import pytest

from trainspotter.cli import main
from trainspotter.detectors import run_all
from trainspotter.readers import detect_format, load_run
from trainspotter.report import render_html
from trainspotter.timing import assess_timing

wandb = pytest.importorskip("wandb")
pb = importlib.import_module("wandb.proto.wandb_internal_pb2")
ROOT = Path(__file__).resolve().parents[1] / "examples/wandb-native"


def history(step: int, items: dict[str, object], *, counter: int | None = None) -> Any:
    record = pb.Record(num=step + 1)
    for name, value in {"_step": step, **items}.items():
        record.history.item.add(nested_key=[name], value_json=json.dumps(value))
    record.history.step.num = step if counter is None else counter
    return record


def native_file(path: Path, records: list[Any]) -> bytes:
    data = bytearray(b":W&B\xe1\xbe\x00")
    for record in records:
        payload = record if isinstance(record, bytes) else record.SerializeToString()
        assert len(data) + len(payload) + 7 < 32768
        data += struct.pack("<IHB", zlib.crc32(b"\x01" + payload) & 0xFFFFFFFF, len(payload), 1)
        data += payload
    path.write_bytes(data)
    return bytes(data)


@pytest.mark.parametrize("case", ["stable", "unstable-lr", "logged-null", "custom-counter"])
def test_sdk_written_capture_retains_every_logged_loss_and_training_coordinate(case: str) -> None:
    directory = ROOT / case
    reference = json.loads((directory / "reference.json").read_text())
    axis = reference["step_key"]
    run = load_run(directory / "run.wandb", step_key=axis)
    series = run.get("train/loss")
    assert series is not None
    expected = [row for row in reference["log_calls"] if "train/loss" in row]
    assert len(series.points) == len(expected) == 80
    for point, row in zip(series.points, expected, strict=True):
        assert point.step == row[axis]
        value = row["train/loss"]
        if value is None:
            assert point.value is None and point.wall_time is not None
        elif math.isnan(float(value)):
            assert point.value is not None and math.isnan(point.value)
        else:
            assert point.value == float(value)
    info = run.meta["wandb_native"]
    assert isinstance(info, dict)
    assert info["history_records"] == 100
    assert info["records_read"] == 311
    assert info["fragments_read"] == 318
    assert info["exit_codes"] == [0]
    assert info["tail_error"] is None
    assert run.input_complete() == (case != "logged-null")


def test_native_custom_chart_axis_does_not_certify_optimizer_update_counts() -> None:
    path = ROOT / "custom-counter/run.wandb"
    run = load_run(path)
    assert detect_format(path) == "wandb-native"
    assert run.axis_kind() == "logging_step"
    assert run.get("train/loss").steps()[-1] == 98
    assert assess_timing(run).status == "unavailable"
    assert (
        run.meta["wandb_native"]["metric_definitions"][1]["definition"]["step_metric"]
        == "optimizer_step"
    )
    selected = load_run(path, step_key="optimizer_step")
    assert selected.axis_kind() == "step"
    assert selected.meta["step_selection"] == "explicit"
    assert selected.get("train/loss").steps()[-1] == 400


def test_real_optimizer_divergence_is_observed_nonfinite_data_not_partial_input() -> None:
    run = load_run(ROOT / "unstable-lr/run.wandb")
    assert run.input_complete()
    assert any(f.detector == "divergence" and f.severity == "error" for f in run_all(run))
    assert any(not math.isfinite(point.value) for point in run.get("train/loss").points)


def test_sdk_empty_first_fragment_at_exact_block_boundary_is_a_complete_record() -> None:
    run = load_run(ROOT / "framing/zero-first.wandb")
    assert run.get("train/loss").steps() == [0]
    assert run.get("train/loss").values() == [1]
    assert run.meta["wandb_native"]["records_read"] == 2
    assert run.meta["wandb_native"]["fragments_read"] == 3
    assert run.input_complete()


@pytest.mark.parametrize("damage", ["truncate", "checksum"])
def test_incomplete_sdk_fragment_chain_cannot_lend_a_partial_training_reading(
    tmp_path: Path,
    damage: str,
) -> None:
    gold = json.loads((ROOT / "fragments.json").read_text())["cases"][0]
    raw = (ROOT / "stable/run.wandb").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == gold["input_sha256"]
    large = next(row for row in gold["large_records"] if row["kind"] == "history")
    if damage == "truncate":
        raw = raw[: large["start"] + 40_000]
    else:
        changed = bytearray(raw)
        changed[large["start"] + 10] ^= 1
        raw = bytes(changed)
    path = tmp_path / "fragment.wandb"
    path.write_bytes(raw)
    run = load_run(path)
    assert run.get("train/loss").steps()[-1] == 95
    assert not run.input_complete()
    assert run.meta["wandb_native"]["unread_from_offset"] == large["start"]


def test_native_reader_never_initializes_a_client_and_reads_read_only_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("reader started a W&B run")

    monkeypatch.setattr(wandb, "init", forbidden)
    path = tmp_path / "copy.wandb"
    data = (ROOT / "stable/run.wandb").read_bytes()
    path.write_bytes(data)
    path.chmod(0o444)
    try:
        assert load_run(path).input_complete()
        assert path.read_bytes() == data
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("damage", ["checksum", "payload", "header"])
def test_bad_tail_retains_only_complete_history_and_cannot_pass_a_ci_gate(
    tmp_path: Path,
    damage: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "tail.wandb"
    first = native_file(path, [history(0, {"train/loss": 1.0})])
    complete = native_file(path, [history(0, {"train/loss": 1.0}), history(1, {"train/loss": 0.5})])
    if damage == "checksum":
        data = bytearray(complete)
        data[-1] ^= 1
        path.write_bytes(data)
    elif damage == "payload":
        path.write_bytes(complete[:-2])
    else:
        path.write_bytes(first + complete[len(first) : len(first) + 3])
    original = path.read_bytes()
    run = load_run(path)
    assert run.get("train/loss").steps() == [0]
    assert not run.input_complete()
    assert run.unread_event_bytes() == len(original) - len(first)
    assert run.meta["wandb_native"]["unread_from_offset"] == len(first)
    assert main(["analyze", str(path), "--output", "json", "--fail-on", "error"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["run"]["wandb_native"]["tail_error"]
    assert not result["run"]["input_complete"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("ambiguous", ["duplicate", "nested", "both", "counter", "json"])
def test_semantically_ambiguous_crc_valid_records_refuse_instead_of_picking_a_value(
    tmp_path: Path,
    ambiguous: str,
) -> None:
    record = history(0, {"train/loss": 1.0})
    if ambiguous == "duplicate":
        record.history.item.add(key="train/loss", value_json="2")
        message = "duplicate native"
    elif ambiguous == "nested":
        record.history.item.add(key="train.loss", value_json="1")
        record.history.item.add(nested_key=["train", "loss"], value_json="2")
        message = "keys collide"
    elif ambiguous == "both":
        record.history.item[1].key = "other"
        message = "key and nested_key"
    elif ambiguous == "counter":
        record.history.step.num = 5
        message = "disagrees"
    else:
        record.history.item[1].value_json = "010"
        message = "invalid native W&B history JSON"
    path = tmp_path / "ambiguous.wandb"
    native_file(path, [record])
    with pytest.raises(ValueError, match=message):
        load_run(path)


def test_two_run_identities_cannot_be_stitched_into_one_trajectory(tmp_path: Path) -> None:
    first, second = pb.Record(), pb.Record()
    first.run.run_id, second.run.run_id = "first", "second"
    path = tmp_path / "mixed.wandb"
    native_file(path, [first, history(0, {"train/loss": 1}), second])
    with pytest.raises(ValueError, match="more than one run identity"):
        load_run(path)


def test_multiple_training_counters_require_a_selection_and_out_of_range_stays_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "axes.wandb"
    native_file(path, [history(0, {"train/loss": 1, "global_step": 10, "train/global_step": 20})])
    with pytest.raises(ValueError, match="choose one with --step-key"):
        load_run(path)
    assert load_run(path, step_key="global_step").get("train/loss").steps() == [10]
    native_file(path, [history(2**53 + 1, {"train/loss": 1}), history(1, {"train/loss": 0.5})])
    run = load_run(path)
    assert run.get("train/loss").steps() == [1]
    assert run.skipped_rows() == 1 and not run.input_complete()


def test_no_missing_axis_or_future_framing_is_filled_with_row_numbers(tmp_path: Path) -> None:
    path = tmp_path / "no-axis.wandb"
    record = pb.Record()
    record.history.item.add(key="train/loss", value_json="1")
    native_file(path, [record])
    with pytest.raises(ValueError, match="no saved logging or training-step"):
        load_run(path)
    raw = bytearray(native_file(path, [history(0, {"train/loss": 1})]))
    raw[6] = 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="version-0"):
        load_run(path)


def test_native_output_cannot_replace_its_input_or_an_inode_alias(tmp_path: Path) -> None:
    path = tmp_path / "source.wandb"
    original = native_file(path, [history(0, {"train/loss": 1})])
    alias = tmp_path / "alias.wandb"
    os.link(path, alias)
    for out in (path, alias):
        assert main(["analyze", str(path), "--output", "json", "--out", str(out)]) == 2
        assert path.read_bytes() == original
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == load_run(path).meta["wandb_native"]["sha256"]
    )


def test_svg_download_retains_native_file_identity_and_partial_read_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.wandb"
    raw = native_file(path, [history(0, {"train/loss": 1}), history(1, {"train/loss": 0.5})])
    path.write_bytes(raw[:-2])
    run = load_run(path)
    report = render_html(run, [])
    encoded = re.search(r'href="data:image/svg\+xml;base64,([^"]+)"', report)
    assert encoded is not None
    svg = ET.fromstring(base64.b64decode(encoded[1]))
    metadata = json.loads(svg.find("{http://www.w3.org/2000/svg}metadata").text)
    assert metadata["sourceFormat"] == "wandb-native" and not metadata["inputComplete"]
    assert metadata["nativeWandb"]["fileSHA256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        metadata["nativeWandb"]["unreadFromOffset"]
        == run.meta["wandb_native"]["unread_from_offset"]
    )
