"""The observations behind a throughput comparison, including unavailable checks.

Derived wall-clock intervals require actual step counts. A row index can locate
an observation, but cannot say how much training happened between observations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from trainspotter.model import AxisKind, MetricSeries, Run

DEFAULT_MIN_WINDOW = 5
DEFAULT_BASELINE_FRAC = 0.2
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TimingWindow:
    first_position: int
    last_position: int
    # For derived timings, the first observation ends an interval beginning here.
    interval_start_step: int | None
    observations: int
    invalid_observations: int
    median_seconds: float | None


@dataclass(slots=True)
class TimingEvidence:
    status: Literal["compared", "unavailable"] = "unavailable"
    axis: AxisKind = "step"
    source: str | None = None
    source_metric: str | None = None
    unit: str | None = None
    observations: int = 0
    invalid_observations: int = 0
    baseline: TimingWindow | None = None
    recent: TimingWindow | None = None
    ratio: float | None = None
    reason: str | None = None
    notes: list[str] = field(default_factory=list)
    min_window: int = DEFAULT_MIN_WINDOW
    baseline_fraction: float = DEFAULT_BASELINE_FRAC

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window(
    steps: list[int], times: FloatArray, starts: list[int] | None,
    lo: int, hi: int,
) -> TimingWindow:
    values = times[lo:hi]
    invalid = int(np.count_nonzero(~np.isfinite(values) | (values <= 0)))
    median = None
    if not invalid:
        ordered = np.sort(values)
        lower = float(ordered[(len(ordered) - 1) // 2])
        upper = float(ordered[len(ordered) // 2])
        # Positive timings can be near the float limit; (lower + upper) / 2 can overflow.
        median = lower + (upper - lower) / 2
    return TimingWindow(steps[lo], steps[hi - 1], starts[lo] if starts else None,
                        hi - lo, invalid, median)


def _compare(
    result: TimingEvidence, run: Run, metric: str, source: str,
    steps: list[int], times: FloatArray, starts: list[int] | None = None,
) -> TimingEvidence:
    result.source = source
    names = run.meta.get("metric_sources", {})
    result.source_metric = str(names.get(metric, metric)) if isinstance(names, dict) else metric
    result.unit = "s/unit" if source == "1/throughput" else "s/step"
    result.observations = len(times)
    result.invalid_observations = int(np.count_nonzero(~np.isfinite(times) | (times <= 0)))
    window = min(max(result.min_window, int(len(times) * min(result.baseline_fraction, 0.5))),
                 len(times) // 2)
    result.baseline = _window(steps, times, starts, 0, window)
    result.recent = _window(steps, times, starts, len(times) - window, len(times))
    early, late = result.baseline.median_seconds, result.recent.median_seconds
    if early is None or late is None:
        result.reason = (
            "The baseline or recent window contains invalid timing observations. "
            "The windows were not moved past them; no speed comparison is available."
        )
        return result
    ratio = late / early
    if not math.isfinite(ratio) or ratio <= 0:
        result.reason = "The timing ratio is outside the representable positive numeric range."
        return result
    result.status = "compared"
    result.ratio = ratio
    if result.invalid_observations:
        result.notes.append(
            f"{result.invalid_observations} invalid timing observations lie outside the two "
            "compared windows; those observations are not evidence of stable speed."
        )
    return result


def _enough(series: MetricSeries | None, required: int, label: str, notes: list[str]) -> bool:
    if series is None:
        return False
    if len(series) >= required:
        return True
    notes.append(f"{label}: {len(series)} analysis points; at least {required} are required.")
    return False


def assess_timing(
    run: Run, *, min_window: int = DEFAULT_MIN_WINDOW,
    baseline_fraction: float = DEFAULT_BASELINE_FRAC,
) -> TimingEvidence:
    """Inspect the same source and windows used by the throughput detector.

    Prefer direct step time, then inverse throughput, then train/eval loss wall
    times. A selected source with broken window values stays unavailable; a
    more convenient lower-priority source must not hide those observations.
    """
    if isinstance(min_window, bool) or not isinstance(min_window, int) or min_window < 1:
        raise ValueError("min_window must be a positive integer")
    if not math.isfinite(baseline_fraction) or baseline_fraction <= 0:
        raise ValueError("baseline_fraction must be finite and positive")
    run = run.analysis_view()
    result = TimingEvidence(min_window=min_window, baseline_fraction=baseline_fraction)
    result.axis = run.axis_kind()
    required = min_window * 2
    step_time = run.get("step_time")
    if _enough(step_time, required, "step_time", result.notes) and step_time is not None:
        return _compare(result, run, "step_time", "step_time", step_time.steps(),
                        np.array(step_time.values(), dtype=float))

    throughput = run.get("throughput")
    if _enough(throughput, required, "throughput", result.notes) and throughput is not None:
        values = np.array(throughput.values(), dtype=float)
        times = np.full(values.shape, np.nan)
        with np.errstate(over="ignore", divide="ignore"):
            np.divide(1.0, values, out=times, where=np.isfinite(values) & (values > 0))
        result.notes.append("A reported unit may be a sample or token; it is not a training step.")
        return _compare(result, run, "throughput", "1/throughput", throughput.steps(), times)

    if result.axis != "step":
        coordinates = ("row indices" if result.axis == "row_index" else "MLflow logged steps"
                       if run.source_format == "mlflow" else "W&B logging steps")
        result.reason = (
            "Wall-clock timing needs an explicit training-step column; this source uses "
            f"{coordinates}. Logging intervals cannot establish seconds per training step. "
            "Log step counts, step_time, or throughput, "
            "or select a training counter with --step-key."
        )
        return result

    for name in ("train/loss", "eval/loss"):
        series = run.get(name)
        if not _enough(series, required + 1, f"wall_time({name})", result.notes):
            continue
        assert series is not None
        missing = sum(p.wall_time is None for p in series.points)
        if missing:
            result.notes.append(
                f"wall_time({name}): {missing} of {len(series)} points have no timestamp. "
                "No intervals were bridged across missing timestamps."
            )
            continue
        wall_times = np.array([p.wall_time for p in series.points], dtype=float)
        steps = series.steps()
        with np.errstate(over="ignore", invalid="ignore"):
            elapsed = np.diff(wall_times)
        advance = np.array([b - a for a, b in zip(steps[:-1], steps[1:], strict=True)], dtype=float)
        times = np.full(elapsed.shape, np.nan)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(elapsed, advance, out=times,
                      where=np.isfinite(elapsed) & (elapsed > 0) & (advance > 0))
        result.notes.append(
            "Wall-clock intervals include evaluation, checkpointing, pauses and other work "
            "between observations. They are not isolated compute timings."
        )
        return _compare(result, run, name, f"wall_time({name})", steps[1:], times, steps[:-1])

    result.reason = (
        "No eligible timing source. The comparison needs enough step_time or throughput "
        "readings, or loss observations with timestamps and explicit training steps. "
        "Conflicting repeated-step values are excluded from this check."
    )
    return result
