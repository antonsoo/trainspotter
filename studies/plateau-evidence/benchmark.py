"""Bound the cost of extra plateau evidence on large, explicitly synthetic histories."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np

from trainspotter.detectors.plateau import PlateauDetector
from trainspotter.model import Run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = []
    for name, values in [
        ("steady-improvement", 10 - np.arange(100_000) * .00005),
        ("noisy-stationary", 5 + np.random.default_rng(42).normal(0, .05, 100_000)),
        ("million-constant", np.full(1_000_000, 5.0)),
    ]:
        run = Run(source_format="synthetic", source_path=name)
        for step, value in enumerate(values):
            run.add_point("train/loss", step, float(value))
        run.finalize()
        start = time.perf_counter()
        findings = PlateauDetector().run(run)
        elapsed = time.perf_counter() - start
        rows.append({"case": name, "observations": len(values), "seconds": elapsed,
                     "findings": len(findings)})
    result = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "available_cpus": len(os.sched_getaffinity(0)),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "timing_scope": "One diagnostic call including analysis projection; excludes Run creation. "
                        "Other verification ran concurrently; these are workload checks, "
                        "not an isolated speed comparison. RSS includes the input Run.",
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
