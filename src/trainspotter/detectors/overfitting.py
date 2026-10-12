"""Evaluation regression, with contemporaneous training evidence when available.

Only the final contiguous finite evaluation segment is assessed. Its lowest
logged value is compared with its last value. The original rise thresholds
are retained: a positive OLS slope with p < 0.1 and >=3% rise, or a >=50%
final rise, or >=25% with the penultimate value >=10% above the minimum.
From 10% the severity is warning; from 50%, error. A >=5% rise with at least
eight later evaluations and slope p < 0.001 is also a warning. Percentages
use the absolute minimum for negative losses. A zero baseline has no relative
rise: an observed increase is informational, with its raw endpoints retained.

The OLS p-values are heuristic diagnostics, not calibrated probabilities of
overfitting: the minimum is selected from the same curve, readings can be
correlated, and the shared slope routine uses a normal approximation.

"Possible overfitting" additionally requires falling training loss AND rising
evaluation loss within their shared step interval, with at least three finite
observations of each. Later training after evaluation ended cannot support
this claim. The exact observed spans are retained; neither curve is interpolated.
Nonfinite training within the shared interval prevents its slope assessment.

A change in evaluation data or metric definition can produce the same pattern.
Aggregate logs cannot establish checkpoint uncertainty or whether a checkpoint
was saved. run_all additionally attributes overlapping divergence/spike findings.
"""

from __future__ import annotations

import math

import numpy as np

from trainspotter.model import Run

from .base import (
    Detector,
    Finding,
    FloatArray,
    Severity,
    finite_spans,
    format_p,
    ols_slope,
    series_arrays,
)


def _relative_rise(best: float, current: float) -> float:
    if best == 0:
        return math.inf if current > 0 else 0.0
    # Subtract after dividing so finite opposite-sign endpoints do not overflow.
    return current / abs(best) - math.copysign(1.0, best)


def _rise(fraction: float, best: float) -> str:
    if best == 0:
        return "relative rise unavailable from a zero baseline"
    if not math.isfinite(fraction):
        return "relative rise exceeds numeric range"
    if best < 0:
        return (
            f"{fraction * 100:.1f}% of the absolute minimum"
            if fraction < 10
            else f"{fraction:.4g} times the absolute minimum higher"
        )
    return f"{fraction * 100:.1f}%" if fraction < 10 else f"{fraction + 1:.4g}x its minimum"


def _scaled_slope(steps: FloatArray, values: FloatArray) -> tuple[float, float, float]:
    scale = float(np.max(np.abs(values)))
    return ols_slope(steps, values / scale if scale else values)


class OverfittingDetector(Detector):
    name = "overfitting"

    def __init__(
        self,
        train_metric: str = "train/loss",
        eval_metric: str = "eval/loss",
        alpha: float = 0.1,
        min_points_after: int = 3,
        min_rise: float = 0.03,
        warn_rise: float = 0.10,
        clear_rise: float = 0.05,
        far_rise: float = 0.25,
        error_rise: float = 0.50,
    ) -> None:
        self.train_metric = train_metric
        self.eval_metric = eval_metric
        self.alpha = alpha
        self.min_points_after = min_points_after
        self.min_rise = min_rise
        self.warn_rise = warn_rise
        self.clear_rise = clear_rise
        self.far_rise = far_rise
        self.error_rise = error_rise

    def _run(self, run: Run) -> list[Finding]:
        eval_series = run.get(self.eval_metric)
        if eval_series is None or len(eval_series) < 3:
            return []
        all_steps, all_values = series_arrays(eval_series)
        spans = finite_spans(all_values)
        # This detector describes the end of the logged run. A final invalid
        # evaluation is reported by divergence, not silently replaced by an earlier one.
        if not spans or spans[-1][1] != len(all_values):
            return []
        start, stop = spans[-1]
        eval_steps, eval_values = all_steps[start:stop], all_values[start:stop]
        if len(eval_values) < 3:
            return []
        best_i = int(np.argmin(eval_values))
        after, after_steps = eval_values[best_i:], eval_steps[best_i:]
        tail = after[1:]
        if len(tail) == 0:
            return []
        best_value, current_value = float(after[0]), float(after[-1])
        rise = _relative_rise(best_value, current_value)
        slope, p_value = 0.0, 1.0
        slope_available = len(tail) >= self.min_points_after and len(after) >= 3
        if slope_available:
            slope, _se, p_value = _scaled_slope(after_steps, after)
        trend = slope > 0 and p_value < self.alpha and rise >= self.min_rise
        far_above = rise >= self.error_rise or (
            rise >= self.far_rise
            and len(tail) >= 2
            and _relative_rise(best_value, float(tail[-2])) >= self.warn_rise
        )
        if not (trend or far_above):
            return []
        train_note, training_evidence = self._training_evidence(run, after_steps, after)
        clear_trend = rise >= self.clear_rise and p_value < 0.001 and len(tail) >= 8
        slope_note = (
            f"slope diagnostic {format_p(p_value)}"
            if trend
            else "rise threshold met; slope criterion not met"
            if slope_available
            else f"over {len(tail)} evaluation{'s' if len(tail) != 1 else ''}, too few for a trend test"
        )
        severity: Severity = (
            "info"
            if best_value == 0
            else "error"
            if rise >= self.error_rise
            else "warning"
            if rise >= self.warn_rise or clear_trend
            else "info"
        )
        segment_note = (
            f" Assessment starts at step {int(eval_steps[0])}, after the last "
            f"{'unreadable' if eval_series.points[start - 1].value is None else 'non-finite'} evaluation."
            if start
            else ""
        )
        return [
            Finding(
                detector=self.name,
                title="Possible overfitting" if train_note else "Evaluation loss rising",
                severity=severity,
                metric=self.eval_metric,
                step_start=int(after_steps[0]),
                step_end=int(after_steps[-1]),
                message=(
                    f"Lowest logged {self.eval_metric} in this finite segment was {best_value:.4g} "
                    f"at step {int(after_steps[0])}; the last value is {current_value:.4g} "
                    f"({_rise(rise, best_value)}, {slope_note})."
                )
                + segment_note
                + train_note,
                evidence={
                    "segment_start": int(eval_steps[0]),
                    "segment_observations": len(eval_values),
                    "best_step": int(after_steps[0]),
                    "best_value": best_value,
                    "current_value": current_value,
                    "relative_rise": rise
                    if math.isfinite(rise)
                    else (
                        "unavailable: zero baseline" if best_value == 0 else "exceeds numeric range"
                    ),
                    "slope_p_value": p_value,
                    "slope_status": "assessed" if slope_available else "insufficient_observations",
                    **training_evidence,
                },
                fixes=[
                    f"If a checkpoint was saved near step {int(after_steps[0])}, compare it with the last saved checkpoint on the same evaluation examples.",
                    "Check evaluation-data and metric-definition changes before attributing the rise to training.",
                    "If the regression persists on a comparable evaluation set, consider early stopping, regularization, or more training data.",
                ],
            )
        ]

    def _training_evidence(
        self, run: Run, eval_steps: FloatArray, eval_values: FloatArray
    ) -> tuple[str, dict[str, float | int | str]]:
        series = run.get(self.train_metric)
        evidence: dict[str, float | int | str] = {"training_status": "missing"}
        if series is None or not len(series):
            return "", evidence
        train_steps, train_values = series_arrays(series)
        lo = max(float(eval_steps[0]), float(train_steps[0]))
        hi = min(float(eval_steps[-1]), float(train_steps[-1]))
        tm = (train_steps >= lo) & (train_steps <= hi)
        em = (eval_steps >= lo) & (eval_steps <= hi)
        evidence.update(
            {
                "training_status": "insufficient_overlap",
                "training_observations": int(tm.sum()),
                "overlap_eval_observations": int(em.sum()),
            }
        )
        if tm.sum() < 3 or em.sum() < 3:
            return "", evidence
        ts, tv = train_steps[tm], train_values[tm]
        es, ev = eval_steps[em], eval_values[em]
        evidence.update(
            {
                "training_step_start": int(ts[0]),
                "training_step_end": int(ts[-1]),
                "overlap_eval_step_start": int(es[0]),
                "overlap_eval_step_end": int(es[-1]),
            }
        )
        if not np.all(np.isfinite(tv)):
            evidence["training_status"] = (
                "unreadable_overlap"
                if any(p.value is None and lo <= p.step <= hi for p in series.points)
                else "nonfinite_overlap"
            )
            return "", evidence
        t_slope, _t_se, t_p = _scaled_slope(ts, tv)
        e_slope, _e_se, e_p = _scaled_slope(es, ev)
        evidence.update(
            {
                "training_status": "opposing_trends_not_established",
                "training_scaled_slope": t_slope,
                "training_slope_p_value": t_p,
                "overlap_eval_scaled_slope": e_slope,
                "overlap_eval_slope_p_value": e_p,
            }
        )
        if t_slope >= 0 or t_p >= 0.2 or e_slope <= 0 or e_p >= self.alpha:
            return "", evidence
        evidence["training_status"] = "opposing_trends"
        return (
            f" Within their shared interval, {self.train_metric} trends down "
            f"({len(ts)} readings, steps {int(ts[0])}-{int(ts[-1])}) while "
            f"{self.eval_metric} trends up ({len(es)} readings, steps {int(es[0])}-{int(es[-1])}). "
            "This is consistent with overfitting, but does not establish its cause."
        ), evidence
