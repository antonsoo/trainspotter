"""Throughput / step-time regression detector.

**Data source, in priority order:** an explicit `step_time` metric
(seconds/step); failing that, `1 / throughput` (throughput = samples or
tokens per second -- inverted so "bigger is slower" like step_time);
failing that, the wall-clock deltas between consecutive points of the
primary loss metric, if wall times were logged. If none of these exist,
the detector reports nothing rather than guessing.

**Algorithm.** Split the derived per-step-time series into an early
baseline window (`baseline_frac`, default first 20%, at least
`min_window`=5 points) and a trailing recent window (same size, last N
points). Take the median of each (median, not mean, so one slow "step"
that includes a checkpoint save doesn't skew the baseline). Flag if
`median(recent) / median(baseline) >= ratio` (default 1.4x).

**False-positive modes.**
- Periodic checkpoint saves, evaluation passes, or logging I/O inflate
  the step time for just that step; a single slow point rarely moves the
  median but a burst of them can.
- Early steps are often slower (CUDA kernel autotuning, dataloader
  warmup, JIT compilation), which can make the *baseline* look
  artificially slow and mask a real regression -- consider excluding the
  first few steps from the baseline window for GPU runs.
- Throughput-based (not step-time-based) input conflates per-sample and
  per-step timing if batch size changed mid-run; a batch-size increase
  alone can look like a "regression" in this metric even though per-step
  work legitimately grew.
"""

from __future__ import annotations

import numpy as np

from trainspotter.model import Run

from .base import Detector, Finding

DEFAULT_RATIO = 1.4
DEFAULT_MIN_WINDOW = 5
DEFAULT_BASELINE_FRAC = 0.2


class ThroughputDetector(Detector):
    name = "throughput"

    def __init__(
        self,
        ratio: float = DEFAULT_RATIO,
        min_window: int = DEFAULT_MIN_WINDOW,
        baseline_frac: float = DEFAULT_BASELINE_FRAC,
    ) -> None:
        self.ratio = ratio
        self.min_window = min_window
        self.baseline_frac = baseline_frac

    def run(self, run: Run) -> list[Finding]:
        steps, times, source = self._derive_step_times(run)
        if steps is None or times is None or len(steps) < self.min_window * 2:
            return []
        n = len(times)
        window = max(self.min_window, int(n * self.baseline_frac))
        if window * 2 > n:
            window = n // 2
        baseline = times[:window]
        recent = times[-window:]
        baseline_med = float(np.median(baseline))
        recent_med = float(np.median(recent))
        if baseline_med <= 0:
            return []
        ratio = recent_med / baseline_med
        if ratio < self.ratio:
            return []
        return [
            Finding(
                detector=self.name,
                title="Throughput regression",
                severity="warning",
                metric=source,
                step_start=int(steps[-window]),
                step_end=int(steps[-1]),
                message=(
                    f"Median step time over the last {window} points ({recent_med:.4g}s) is "
                    f"{ratio:.2f}x the early-run baseline ({baseline_med:.4g}s, first {window} "
                    f"points), derived from {source}."
                ),
                evidence={
                    "baseline_median_s": baseline_med,
                    "recent_median_s": recent_med,
                    "ratio": ratio,
                    "window": window,
                    "source": source,
                },
                fixes=[
                    "Check for a dataloader/IO bottleneck (e.g. a slow shard, network storage stall, growing prefetch queue).",
                    "Check for memory pressure/fragmentation causing more frequent allocator work or CPU-GPU sync.",
                    "Check other processes on the same GPU/host, or a thermal/power throttling event.",
                    "If a sequence-length or batch-size curriculum increases work per step by design, this may be expected -- not a regression.",
                ],
            )
        ]

    def _derive_step_times(
        self, run: Run
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        step_time = run.get("step_time")
        if step_time is not None and len(step_time) >= self.min_window * 2:
            steps = np.array(step_time.steps(), dtype=float)
            times = np.array(step_time.values(), dtype=float)
            return steps, times, "step_time"

        throughput = run.get("throughput")
        if throughput is not None and len(throughput) >= self.min_window * 2:
            steps = np.array(throughput.steps(), dtype=float)
            vals = np.array(throughput.values(), dtype=float)
            vals = np.where(vals > 0, 1.0 / vals, np.nan)
            valid = ~np.isnan(vals)
            if valid.sum() >= self.min_window * 2:
                return steps[valid], vals[valid], "1/throughput"

        for name in ("train/loss", "eval/loss"):
            series = run.get(name)
            if series is None:
                continue
            wall_times = [p.wall_time for p in series.points]
            if any(w is None for w in wall_times) or len(wall_times) < self.min_window * 2 + 1:
                continue
            wt = np.array(wall_times, dtype=float)
            st = np.array(series.steps(), dtype=float)
            dt = np.diff(wt)
            positive = dt > 0
            if positive.sum() < self.min_window * 2:
                continue
            return st[1:][positive], dt[positive], f"wall_time({name})"

        return None, None, ""
