"""Regression test over the real committed example runs: trust matters as
much as recall. A tool that flags an unremarkable, healthy run gets
ignored, so `baseline` -- ordinary warmup + cosine-decay training, nothing
wrong with it -- is asserted to come back essentially clean, while the
other four examples must still flag the pathology they were built to
demonstrate. Found via this exact gap: baseline used to produce 4 plateau
infos and a grad-norm warning before the plateau/grad_norm detectors were
tightened (see their docstrings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trainspotter.detectors import Finding, Severity, run_all, severity_at_least
from trainspotter.readers import load_run

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

_SEVERITY_RANK: dict[Severity, int] = {"error": 0, "warning": 1, "info": 2}


def _findings_for(name: str) -> list[Finding]:
    run = load_run(EXAMPLES_DIR / f"{name}.trainer_state.json")
    return run_all(run)


def test_baseline_is_essentially_clean() -> None:
    findings = _findings_for("baseline")
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    infos = [f for f in findings if f.severity == "info"]

    assert errors == []
    assert warnings == []
    assert len(infos) <= 1


def test_overfitting_example_flags_overfitting() -> None:
    findings = _findings_for("overfitting")
    assert any(f.detector == "overfitting" for f in findings)


def test_divergence_example_flags_divergence_and_a_spike() -> None:
    findings = _findings_for("divergence")
    assert any(f.detector == "divergence" and f.severity == "error" for f in findings)
    assert any(f.detector == "spikes" for f in findings)


def test_missing_warmup_example_flags_missing_warmup() -> None:
    findings = _findings_for("missing_warmup")
    assert any(f.detector == "lr_schedule" and "warmup" in f.title.lower() for f in findings)


def test_throughput_drop_example_flags_throughput_regression() -> None:
    findings = _findings_for("throughput_drop")
    assert any(f.detector == "throughput" for f in findings)


@pytest.mark.parametrize(
    "name", ["baseline", "overfitting", "divergence", "missing_warmup", "throughput_drop"]
)
def test_findings_are_ordered_by_severity_then_step(name: str) -> None:
    findings = _findings_for(name)
    ranks = [(_SEVERITY_RANK[f.severity], f.step_start) for f in findings]
    assert ranks == sorted(ranks)
    # sanity check on severity_at_least, used by --fail-on: errors are at
    # least as severe as warnings, which are at least as severe as info.
    assert severity_at_least("error", "warning")
    assert not severity_at_least("info", "warning")
