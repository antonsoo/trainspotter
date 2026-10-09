"""Throughput / step-time regression detector.

**Data source, in priority order:** an explicit `step_time` metric
(seconds/step); failing that, `1 / throughput` (throughput = samples or
tokens per second -- inverted so "bigger is slower" like step_time);
failing that, the wall-clock deltas between consecutive points of the
primary loss metric, if wall times were logged. If none of these exist,
the detector reports nothing rather than guessing.
Wall-clock derivation also needs explicit training steps: a row index cannot
distinguish slow training from less frequent logging. Reports expose timing
source, windows and unavailable reasons through `trainspotter.timing`.

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

from trainspotter.model import Run
from trainspotter.timing import DEFAULT_BASELINE_FRAC, DEFAULT_MIN_WINDOW, assess_timing

from .base import Detector, Finding

DEFAULT_RATIO = 1.4


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

    def _run(self, run: Run) -> list[Finding]:
        timing = assess_timing(run, min_window=self.min_window, baseline_fraction=self.baseline_frac)
        if timing.ratio is None or timing.ratio < self.ratio:
            return []
        assert timing.baseline is not None and timing.recent is not None
        assert timing.source is not None and timing.unit is not None
        assert timing.baseline.median_seconds is not None and timing.recent.median_seconds is not None
        baseline_med, recent_med = timing.baseline.median_seconds, timing.recent.median_seconds
        ratio, source, unit = timing.ratio, timing.source, timing.unit
        window = timing.recent.observations
        quantity = "time per reported unit" if source == "1/throughput" else "step time"
        return [
            Finding(
                detector=self.name,
                title="Throughput regression",
                severity="warning",
                metric=source,
                step_start=timing.recent.first_position,
                step_end=timing.recent.last_position,
                message=(
                    f"Median {quantity} over the last {window} points ({recent_med:.4g} {unit}) is "
                    f"{ratio:.2f}x the early-run baseline ({baseline_med:.4g} {unit}, first {window} "
                    f"points), derived from {source}."
                ),
                evidence={
                    "baseline_median_s": baseline_med,
                    "recent_median_s": recent_med,
                    "ratio": ratio,
                    "window": window,
                    "source": source,
                    "unit": unit,
                },
                fixes=[
                    "Check for a dataloader/IO bottleneck (e.g. a slow shard, network storage stall, growing prefetch queue).",
                    "Check for memory pressure/fragmentation causing more frequent allocator work or CPU-GPU sync.",
                    "Check other processes on the same GPU/host, or a thermal/power throttling event.",
                    "If a sequence-length or batch-size curriculum increases work per step by design, this may be expected -- not a regression.",
                ],
            )
        ]
