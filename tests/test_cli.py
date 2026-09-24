"""End-to-end CLI tests: invoke `trainspotter.cli.main` the way a shell would,
against real files, and check exit codes and output shape."""

from __future__ import annotations

import json
from pathlib import Path

from trainspotter.cli import main


def _write_diverging_log(path: Path) -> None:
    log_history = []
    loss = 2.5
    for step in range(1, 61):
        loss *= 1.08  # runaway growth -> should trip the divergence detector
        log_history.append({"step": step, "loss": loss, "learning_rate": 1e-3, "epoch": step / 60})
    path.write_text(json.dumps({"log_history": log_history}))


def _write_clean_log(path: Path) -> None:
    # Long enough, and with a long enough warmup relative to the whole run,
    # that no detector's threshold is marginal: a genuinely unremarkable run.
    log_history = []
    warmup_steps = 15
    total_steps = 150
    peak_lr = 1e-4
    for step in range(1, total_steps + 1):
        loss = 2.5 - 0.01 * step  # constant-rate decay: never plateaus, never spikes
        if step <= warmup_steps:
            lr = peak_lr * step / warmup_steps
        else:
            lr = peak_lr * 0.995 ** (step - warmup_steps)
        log_history.append(
            {"step": step, "loss": loss, "learning_rate": lr, "epoch": step / total_steps}
        )
    path.write_text(json.dumps({"log_history": log_history}))


def test_analyze_terminal_exit_code_clean(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trainer_state.json"
    _write_clean_log(path)

    code = main(["analyze", str(path), "--fail-on", "warning"])

    assert code == 0
    out = capsys.readouterr().out
    assert "No pathologies detected" in out


def test_analyze_fails_on_diverging_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trainer_state.json"
    _write_diverging_log(path)

    code = main(["analyze", str(path), "--fail-on", "error"])

    assert code == 1
    out = capsys.readouterr().out
    assert "diverg" in out.lower()


def test_analyze_json_output_is_valid(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trainer_state.json"
    _write_diverging_log(path)

    code = main(["analyze", str(path), "--output", "json"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 0  # --fail-on not passed, so json output alone doesn't fail the build
    assert payload["schema_version"] == 1
    assert payload["summary"]["total_findings"] >= 1
    assert any(f["severity"] == "error" for f in payload["findings"])


def test_analyze_html_output_is_self_contained(tmp_path: Path) -> None:
    path = tmp_path / "trainer_state.json"
    _write_diverging_log(path)
    out_path = tmp_path / "report.html"

    code = main(["analyze", str(path), "--output", "html", "--out", str(out_path)])

    assert code == 0
    html = out_path.read_text()
    assert "<svg" in html
    assert "http://" not in html and "https://" not in html  # no CDN/external requests
    assert "<script" not in html  # no JS needed


def test_watch_once_reports_findings(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trainer_state.json"
    _write_diverging_log(path)

    code = main(["watch", str(path), "--once", "--fail-on", "error"])

    assert code == 1
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_analyze_missing_file_reports_a_clean_error(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code = main(["analyze", str(tmp_path / "nope.json")])

    assert code == 2
    assert "nope.json" in capsys.readouterr().err
