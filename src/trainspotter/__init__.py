"""trainspotter: automatic diagnosis of loss curves and training logs."""

from __future__ import annotations

from trainspotter.detectors import Finding, default_detectors, run_all
from trainspotter.model import MetricSeries, Point, Run
from trainspotter.readers import load_run

__version__ = "0.2.4"

__all__ = [
    "__version__",
    "Run",
    "MetricSeries",
    "Point",
    "Finding",
    "load_run",
    "run_all",
    "default_detectors",
]
