"""Known stationary levels with controlled observation noise, not training accuracy labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trainspotter.detectors.plateau import PlateauDetector
from trainspotter.model import Run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for noise in [0, .001, .005, .01, .02, .05, .1, .2]:
        counts, coverage = [], []
        for seed in range(args.seed_start, args.seed_start + 30):
            rng = np.random.default_rng(seed)
            values = np.r_[
                10 * np.exp(-np.arange(100) / 20) + 5,
                5 * (1 + rng.normal(0, noise, 500)),
                5 * np.exp(-np.arange(100) / 20) + .5,
            ]
            run = Run(source_format="synthetic", source_path="known-stationary-level")
            for step, value in enumerate(values):
                run.add_point("train/loss", step, float(value))
            run.finalize()
            findings = PlateauDetector().run(run)
            covered = {i for finding in findings
                       for i in range(max(100, finding.step_start), min(600, finding.step_end + 1))}
            counts.append(len(findings))
            coverage.append(len(covered) / 500)
        rows.append({"relative_noise_sd": noise, "seeds": 30,
                     "runs_with_a_finding": sum(count > 0 for count in counts),
                     "median_findings": float(np.median(counts)), "max_findings": max(counts),
                     "mean_stationary_region_coverage": float(np.mean(coverage))})
    result = {
        "kind": "synthetic sensitivity sweep with known stationary expected loss",
        "numpy": np.__version__,
        "seeds": f"numpy default_rng({args.seed_start}..{args.seed_start + 29})",
        "shape": "100-point decay; 500 independent Gaussian readings with mean 5; 100-point decay",
        "oracle": "Expected loss is exactly 5 at steps 100-599; later improvement makes a stall",
        "limits": "IID Gaussian observation noise is not a model of all training logs. "
                  "The current guard may miss noisy stationary levels and split a long plateau.",
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
