"""Watch lifecycle and CI/report behavior exercised through the actual CLI."""

from __future__ import annotations

import json

import pytest

from trainspotter import cli


def _polls(monkeypatch, actions):
    remaining = iter(actions)

    def sleep(_seconds):
        action = next(remaining, None)
        if action is None:
            raise KeyboardInterrupt
        action()

    monkeypatch.setattr(cli.time, "sleep", sleep)


def _loss(path, values):
    path.write_text("\n".join(json.dumps({"step": i, "loss": v}) for i, v in enumerate(values)))


@pytest.mark.parametrize("text", [None, "", "{broken", '{"log_history":[]}'])
def test_watch_once_ends_with_an_error_for_unreadable_input(tmp_path, monkeypatch, text):
    path = tmp_path / "run.json"
    if text is not None:
        path.write_text(text)
    monkeypatch.setattr(cli.time, "sleep", lambda _: pytest.fail("--once must never wait"))
    assert cli.main(["watch", str(path), "--once"]) == 2


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "-inf", "1e400", "1e300"])
def test_watch_rejects_invalid_intervals(tmp_path, interval):
    with pytest.raises(SystemExit) as exc:
        cli.main(["watch", str(tmp_path / "run.json"), "--interval=" + interval, "--once"])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["analyze", "watch"])
def test_a_configured_gate_cannot_pass_partial_input(tmp_path, capsys, command):
    path = tmp_path / "run.jsonl"
    path.write_text('{"step":0,"loss":1}\n{broken\n')
    args = [command, str(path), "--fail-on", "error"]
    args += ["--once"] if command == "watch" else ["--output", "json"]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert "input is incomplete" in captured.err
    if command == "analyze":
        report = json.loads(captured.out)
        assert not report["summary"]["input_complete"]
        assert report["run"]["skipped_rows"] == 1
        assert report["findings"] == []


def test_a_one_point_nonfinite_loss_fails_the_gate(tmp_path):
    path = tmp_path / "run.jsonl"
    _loss(path, [float("nan")])
    assert cli.main(["analyze", str(path), "--fail-on", "error", "--output", "json"]) == 1


def test_watch_recovers_after_missing_and_invalid_polls(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    _polls(monkeypatch, [lambda: path.write_text("{partial"), lambda: _loss(path, [1.0])])
    assert cli.main(["watch", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("skipped this read") == 2
    assert "Read recovered" in out


def test_watch_does_not_repeat_an_extending_nonfinite_range(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    _loss(path, [1.0, float("nan")])
    _polls(monkeypatch, [lambda: _loss(path, [1.0, float("nan"), float("nan")]),
                        lambda: None])
    assert cli.main(["watch", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("Diverged (non-finite values)") == 1
    assert "history " not in out


def test_watch_rechecks_rewritten_history(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    _loss(path, [1.0, float("nan"), 2.0])
    _polls(monkeypatch, [lambda: _loss(path, [9.0, float("nan")])])
    assert cli.main(["watch", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("Diverged (non-finite values)") == 2
    assert "Previously read data changed" in out


def test_watch_reports_each_new_lr_rise_once(tmp_path, monkeypatch, capsys):
    path = tmp_path / "lr.csv"
    values = [0.1, 0.05, 0.01]

    def write():
        path.write_text("step,learning_rate\n" + "".join(
            f"{i},{v}\n" for i, v in enumerate(values)
        ))

    def append(more):
        values.extend(more)
        write()

    write()
    _polls(monkeypatch, [lambda: append([0.02]), lambda: append([0.03, 0.04]),
                         lambda: append([0.001, 0.02]), lambda: append([0.03])])
    assert cli.main(["watch", str(path), "--fail-on", "warning"]) == 0
    out = capsys.readouterr().out
    assert out.count("LR rose after decay") == 2
    assert "steps 2-3" in out and "steps 6-7" in out


def test_watch_keeps_the_selected_run_when_another_grows_larger(tmp_path, monkeypatch, capsys):
    path = tmp_path / "chart.csv"
    header = "Step,a - loss,a - loss__MIN,a - loss__MAX,b - loss,b - loss__MIN,b - loss__MAX\n"
    path.write_text(header + "0,2,2,2,,,\n")
    _polls(monkeypatch, [lambda: path.write_text(
        header + "0,2,2,2,nan,nan,nan\n1,,,,nan,nan,nan\n2,,,,nan,nan,nan\n",
    )])
    assert cli.main(["watch", str(path), "--fail-on", "error"]) == 0
    out = capsys.readouterr().out
    assert "a (selection held while watching)" in out
    assert "Diverged" not in out


def test_watch_does_not_switch_when_the_selected_run_disappears(tmp_path, monkeypatch, capsys):
    path = tmp_path / "chart.csv"
    path.write_text("Step,a - loss,a - loss__MIN,a - loss__MAX\n0,2,2,2\n")
    _polls(monkeypatch, [lambda: path.write_text(
        "Step,b - loss,b - loss__MIN,b - loss__MAX\n0,nan,nan,nan\n",
    )])
    assert cli.main(["watch", str(path)]) == 2
    out = capsys.readouterr().out
    assert "matches no run" in out
    assert "Diverged" not in out


@pytest.mark.parametrize("output", ["terminal", "json", "html"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_a_report_cannot_overwrite_its_input(tmp_path, output, alias):
    path = tmp_path / "run.jsonl"
    _loss(path, [1.0])
    original = path.read_bytes()
    target = path
    if alias != "direct":
        target = tmp_path / "output"
        if alias == "symlink":
            target.symlink_to(path)
        else:
            target.hardlink_to(path)
    assert cli.main(["analyze", str(path), "--output", output, "--out", str(target)]) == 2
    assert path.read_bytes() == original


def test_failed_export_preserves_the_previous_report(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    _loss(path, [1.0])
    target = tmp_path / "report.json"
    target.write_text("previous report")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(cli.os, "replace", fail)
    assert cli.main(["analyze", str(path), "--output", "json", "--out", str(target)]) == 2
    assert target.read_text() == "previous report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "run.jsonl"]


def test_terminal_out_writes_the_file_and_does_not_mutate_color_environment(
    tmp_path, monkeypatch, capsys,
):
    path = tmp_path / "run.jsonl"
    _loss(path, [1.0])
    target = tmp_path / "report.txt"
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli.main(["analyze", str(path), "--out", str(target), "--no-color"]) == 0
    assert "trainspotter" in target.read_text()
    assert "\x1b" not in target.read_text()
    assert capsys.readouterr().out == ""
    assert "NO_COLOR" not in cli.os.environ


def test_names_cannot_forge_terminal_status_lines(tmp_path, capsys):
    path = tmp_path / "run\nERROR\x1b.jsonl"
    _loss(path, [1.0])
    assert cli.main(["watch", str(path), "--once"]) == 0
    out = capsys.readouterr().out
    assert "run\\x0aERROR\\x1b.jsonl" in out
    assert "\nERROR" not in out and "\x1b" not in out


def test_an_unterminated_csv_quote_is_a_clean_input_error(tmp_path, capsys):
    path = tmp_path / "run.csv"
    path.write_text('step,loss\n1,2\n2,"3\n')
    assert cli.main(["analyze", str(path), "--fail-on", "error"]) == 2
    captured = capsys.readouterr()
    assert "trainspotter:" in captured.err
    assert "Traceback" not in captured.err
