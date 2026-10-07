"""Real event files, compared with TensorBoard's independent accumulator."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from trainspotter.readers import load_run

event_pb2 = pytest.importorskip("tensorboard.compat.proto.event_pb2")
summary_pb2 = pytest.importorskip("tensorboard.compat.proto.summary_pb2")
writer_module = pytest.importorskip("tensorboard.summary.writer.event_file_writer")
tensor_util = pytest.importorskip("tensorboard.util.tensor_util")
scalar_metadata = pytest.importorskip("tensorboard.plugins.scalar.metadata")
accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")


def _point(step, value, tag="loss", *, tensor=False, wall=None):
    fields = {"tag": tag}
    if tensor:
        fields["tensor"] = tensor_util.make_tensor_proto(value)
        fields["metadata"] = scalar_metadata.create_summary_metadata("", "")
    else:
        fields["simple_value"] = value
    return event_pb2.Event(
        step=step, wall_time=step + 100 if wall is None else wall,
        summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(**fields)]),
    )


def _restart(step):
    return event_pb2.Event(
        step=step, wall_time=10,
        session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START),
    )


def _write(directory, events, *, version=2):
    existing = set(directory.glob("*tfevents.*"))
    writer = writer_module.EventFileWriter(str(directory))
    writer.add_event(event_pb2.Event(file_version=f"brain.Event:{version}"))
    for event in events:
        writer.add_event(event)
    writer.close()
    created = next(iter(set(directory.glob("*tfevents.*")) - existing))
    # EventFileWriter's unpadded process counter can sort .10 before .9 within
    # the same second. Pin filenames so these tests encode the intended order.
    created.rename(directory / f"events.out.tfevents.{len(existing):010d}.test")


def _oracle(directory, tag, *, tensor=False):
    if tensor:
        # The legacy accumulator does not purge its tensor reservoir. Compare the
        # same events encoded as legacy scalars, without changing order or restart
        # markers. All values used by these oracle tests are exactly float32.
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

        with tempfile.TemporaryDirectory() as tmp:
            for path in sorted(directory.glob("*tfevents.*")):
                events = list(LegacyEventFileLoader(str(path)).Load())
                for event in events:
                    for value in event.summary.value:
                        if value.HasField("tensor"):
                            number = float(tensor_util.make_ndarray(value.tensor).item())
                            value.ClearField("tensor")
                            value.simple_value = number
                _write(Path(tmp), events)
            return _oracle(Path(tmp), tag)
    reader = accumulator.EventAccumulator(
        str(directory), size_guidance={"scalars": 0, "tensors": 0},
    ).Reload()
    events = reader.Tensors(tag) if tensor else reader.Scalars(tag)
    return sorted(
        [(event.step, float(tensor_util.make_ndarray(event.tensor_proto).item())
          if tensor else event.value) for event in events], key=lambda pair: pair[0],
    )


def _actual(run, metric):
    series = run.get(metric)
    return [] if series is None else list(zip(series.steps(), series.values(), strict=True))


@pytest.mark.parametrize("tensor", [False, True])
def test_out_of_order_events_and_backward_clocks_do_not_erase_modern_history(
    tmp_path: Path, tensor: bool,
) -> None:
    events = [_point(0, 3, tensor=tensor, wall=100), _point(10, 99, tensor=tensor, wall=90),
              _point(5, 2, tensor=tensor, wall=110)]
    _write(tmp_path, events)
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == _oracle(tmp_path, "loss", tensor=tensor)
    assert _actual(run, "train/loss") == [(0, 3), (5, 2), (10, 99)]
    assert run.overwritten_points() == 0


@pytest.mark.parametrize("tensor", [False, True])
def test_restart_in_another_file_purges_all_tags_even_if_not_logged_again(
    tmp_path: Path, tensor: bool,
) -> None:
    _write(tmp_path, [_point(0, 3, tensor=tensor), _point(10, 99, tensor=tensor),
                      _point(10, 88, "eval_loss", tensor=tensor)])
    _write(tmp_path, [_restart(5), _point(5, 2, tensor=tensor, wall=1)])
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == _oracle(tmp_path, "loss", tensor=tensor)
    assert _actual(run, "eval/loss") == _oracle(tmp_path, "eval_loss", tensor=tensor) == []
    assert run.overwritten_points() == 2
    assert run.meta["restart_markers"] == 1
    assert run.input_complete()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("tensor", [False, True])
def test_seeded_event_histories_match_tensorboard_oracle(
    tmp_path: Path, version: int, tensor: bool,
) -> None:
    rng = random.Random(712)
    tags = ["train/loss", "eval/loss", "grad_norm"]
    for _ in range(3):
        events = []
        for i in range(80):
            step = rng.randrange(40)
            if rng.random() < 0.1:
                events.append(_restart(step))
            events.append(_point(step, i + 1.0, rng.choice(tags), tensor=tensor,
                                 wall=rng.randrange(100)))
        _write(tmp_path, events, version=version)
    run = load_run(tmp_path)
    for tag in tags:
        assert _actual(run, tag) == _oracle(tmp_path, tag, tensor=tensor)


def test_scalar_encodings_can_change_without_losing_event_order(tmp_path: Path) -> None:
    _write(tmp_path, [_point(1, 2), _point(2, 3, tensor=True), _restart(2),
                      _point(2, 4, tensor=True), _point(3, 5)])
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == [(1, 2), (2, 4), (3, 5)]
    assert run.overwritten_points() == 1


def test_scalar_tensor_shape_is_checked_before_allocation(tmp_path: Path) -> None:
    vector = _point(1, [1.0, 2.0], "bad", tensor=True)
    huge = _point(2, [1.0], "huge", tensor=True)
    huge.summary.value[0].tensor.tensor_shape.dim[0].size = 2**60
    _write(tmp_path, [_point(0, 2, tensor=True), vector, huge])
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == [(0, 2)]
    assert run.skipped_rows() == 2
    assert not run.input_complete()


def test_tensor_scalar_nan_is_kept_for_the_divergence_detector(tmp_path: Path) -> None:
    from trainspotter.detectors import run_all

    _write(tmp_path, [_point(i, 1.0 if i < 40 else float("nan"), tensor=True) for i in range(50)])
    run = load_run(tmp_path)
    found = [finding for finding in run_all(run) if finding.detector == "divergence"]
    assert found and found[0].step_start == 40
    assert run.input_complete()


def test_no_reservoir_sampling_drops_a_brief_incident(tmp_path: Path) -> None:
    _write(tmp_path, [_point(i, 99 if i == 5432 else 1, tensor=True) for i in range(10_050)])
    run = load_run(tmp_path)
    series = run.metrics["train/loss"]
    assert len(series) == 10_050
    assert series.points[5432].value == 99


def test_truncated_tail_is_visible_and_recovers_when_completed(tmp_path: Path) -> None:
    _write(tmp_path, [_point(1, 2), _point(2, 3)])
    path = next(tmp_path.glob("*tfevents*"))
    complete = path.read_bytes()
    path.write_bytes(complete[:-3])
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == [(1, 2)]
    assert run.unread_event_bytes() > 0
    assert not run.input_complete()
    path.write_bytes(complete)
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == [(1, 2), (2, 3)]
    assert run.input_complete()


def test_restart_scope_is_one_writer_directory(tmp_path: Path) -> None:
    _write(tmp_path / "train", [_point(0, 3), _point(10, 99), _restart(5), _point(5, 2)])
    _write(tmp_path / "validation", [_point(10, 4)])
    run = load_run(tmp_path)
    assert _actual(run, "train/loss") == [(0, 3), (5, 2)]
    assert _actual(run, "eval/loss") == [(10, 4)]


def test_separate_writers_with_the_same_logical_metric_do_not_merge(tmp_path: Path) -> None:
    _write(tmp_path, [_point(1, 2, "train/loss")])
    _write(tmp_path / "train", [_point(1, 99, "loss")])
    run = load_run(tmp_path)
    assert len(run.metrics) == 2
    assert sorted(series.values()[0] for series in run.metrics.values()) == [2, 99]
    assert len(set(run.meta["metric_sources"].values())) == 2
    assert not run.input_complete()
    assert "multiple writer directories" in " ".join(run.input_notes())


def test_default_run_uses_retained_points_not_discarded_history(tmp_path: Path) -> None:
    _write(tmp_path / "old", [_point(i, 1) for i in range(100)] + [_restart(1), _point(1, 2)])
    _write(tmp_path / "current", [_point(i, 3) for i in range(10)])
    run = load_run(tmp_path)
    assert run.meta["selected_run"] == "current"
    assert len(run.metrics["train/loss"]) == 10


def test_selected_missing_run_is_rejected_even_for_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no runs"):
        load_run(tmp_path, run_name="missing")
