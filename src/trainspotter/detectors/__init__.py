"""Detector registry: every pathology check trainspotter knows, run together."""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding, Severity, severity_at_least
from .divergence import DivergenceDetector
from .eval_noise import EvalNoiseDetector
from .grad_norm import GradNormDetector
from .loss_floor import LossFloorDetector
from .lr_schedule import LRScheduleDetector
from .overfitting import OverfittingDetector
from .plateau import PlateauDetector
from .setbacks import SetbackDetector
from .spikes import SpikeDetector
from .throughput import ThroughputDetector

__all__ = [
    "Detector",
    "Finding",
    "Severity",
    "severity_at_least",
    "default_detectors",
    "run_all",
]


def _step_span(finding: Finding) -> str:
    if finding.step_start == finding.step_end:
        return f"step {finding.step_start}"
    return f"steps {finding.step_start}-{finding.step_end}"


def default_detectors() -> list[Detector]:
    """The standard detector set, run against trainspotter's canonical
    metric names. A detector that finds none of its required metrics in
    the run simply returns no findings -- it's safe to always run all of
    them."""
    return [
        SpikeDetector(metrics=("train/loss", "eval/loss")),
        SetbackDetector(),
        DivergenceDetector(),
        PlateauDetector(),
        OverfittingDetector(),
        LRScheduleDetector(),
        GradNormDetector(),
        ThroughputDetector(),
        EvalNoiseDetector(),
        LossFloorDetector(),
    ]


def run_all(run: Run, detectors: list[Detector] | None = None) -> list[Finding]:
    """Run every detector against `run` and return findings sorted by
    (severity, step_start) -- errors first, then warnings, then info, so
    the most important thing a run did wrong is always what a reader sees
    first; earliest-first within a severity level. (The HTML report's
    health strip stays chronological regardless -- it's a timeline, not a
    priority list.)

    Detectors run independently against `run`, not against each other's
    output, with one deliberate exception applied here afterward: see
    `_attribute_overfitting_to_divergence`."""
    detectors = detectors if detectors is not None else default_detectors()
    findings: list[Finding] = []
    analysis = run.analysis_view()
    # A diverged run logs Inf and NaN, and the detectors are written to take them. numpy
    # would report each operation that touches one ("invalid value encountered in
    # subtract") on stderr, above the report that already says the run diverged.
    with np.errstate(all="ignore"):
        for detector in detectors:
            findings.extend(detector.run(analysis))
    _attribute_overfitting_to_divergence(findings)
    rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (rank[f.severity], f.step_start))
    return findings


def _attribute_overfitting_to_divergence(findings: list[Finding]) -> None:
    """A run that diverges also, trivially, satisfies `overfitting`'s test:
    eval loss shooting to infinity is technically "eval getting worse".
    That's not a second, independent problem -- it's the same explosion the
    `divergence`/`spikes` findings already describe, and reporting it as an
    equally-weighted "overfitting" finding both misattributes the cause and
    doubles up the alarm.

    If a `divergence` finding, or a `spikes` finding on the training loss,
    starts inside an `overfitting` finding's step range or within that
    range's own length after it (the training loss is usually logged just
    after the evaluation that first shows the damage), the `overfitting`
    finding is downgraded to `info` and its message notes why -- mutates
    `findings` in place, called once by `run_all` after every detector has
    run, since only then do both step ranges exist to compare."""
    culprits = [
        f
        for f in findings
        if f.detector == "divergence" or (f.detector == "spikes" and f.metric == "train/loss")
    ]
    if not culprits:
        return
    for f in findings:
        if f.detector != "overfitting" or f.severity == "info":
            continue
        slack = max(1, f.step_end - f.step_start)
        overlap = next(
            (c for c in culprits if f.step_start <= c.step_end and c.step_start <= f.step_end + slack),
            None,
        )
        if overlap is None:
            continue
        f.severity = "info"
        f.message += (
            f" (Downgraded: a {overlap.detector} finding at {_step_span(overlap)} explains "
            "this rise better than ordinary overfitting does.)"
        )
