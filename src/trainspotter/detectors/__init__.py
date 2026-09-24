"""Detector registry: every pathology check trainspotter knows, run together."""

from __future__ import annotations

from trainspotter.model import Run

from .base import Detector, Finding, Severity, severity_at_least
from .divergence import DivergenceDetector
from .eval_noise import EvalNoiseDetector
from .grad_norm import GradNormDetector
from .loss_floor import LossFloorDetector
from .lr_schedule import LRScheduleDetector
from .overfitting import OverfittingDetector
from .plateau import PlateauDetector
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


def default_detectors() -> list[Detector]:
    """The standard detector set, run against trainspotter's canonical
    metric names. A detector that finds none of its required metrics in
    the run simply returns no findings -- it's safe to always run all of
    them."""
    return [
        SpikeDetector(metrics=("train/loss", "eval/loss")),
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
    (step_start, severity desc) -- earliest problems first, worst first
    within a step."""
    detectors = detectors if detectors is not None else default_detectors()
    findings: list[Finding] = []
    for detector in detectors:
        findings.extend(detector.run(run))
    rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (f.step_start, rank[f.severity]))
    return findings
