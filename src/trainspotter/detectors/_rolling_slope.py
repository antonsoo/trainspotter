"""Window slope decisions with bounded prefix sums and direct-fit fallbacks.

The plateau detector needs a decision, not the fitted coefficients. For a
nonconstant window, its normal-approximation test is equivalent to
    Sxy**2 <= c * Sxx * Syy,
where c = z**2 / (n - 2 + z**2), z = Phi^-1(alpha / 2), and S are centered
sums. This avoids subtracting two nearly equal sums to estimate the residual
variance of a very straight line. Prefix sums supply all overlapping windows
in linear work. Rebase them in local blocks so cancellation does not grow with
the length of the complete history.

Cancellation-prone windows and decisions near the threshold use the direct
OLS fit. That fallback is intentional: performance must not decide whether a
stall gets reported. The common path is O(n); the fallback can be O(n*window).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from .base import FloatArray, ols_slope

BoolArray = npt.NDArray[np.bool_]


def _window_sums(values: FloatArray, window: int) -> tuple[FloatArray, float]:
    prefix = np.r_[0.0, np.cumsum(values)]
    sums = prefix[window:] - prefix[:-window]
    # Sequential addition, subtraction of prefixes, and later moment arithmetic.
    # Deliberately wider than the first-order n*eps bound; if this is too loose,
    # we pay for a direct fit rather than accepting an uncertain decision.
    error = 16 * np.finfo(float).eps * len(values) * float(np.sum(np.abs(values)))
    return sums, error


def _constant_windows(values: FloatArray, window: int) -> BoolArray:
    changes = np.r_[0, np.cumsum(values[1:] != values[:-1])]
    result: BoolArray = changes[window - 1 :] == changes[: len(values) - window + 1]
    return result


def flat_slope_windows(
    x: FloatArray, y: FloatArray, window: int, alpha: float, candidates: BoolArray
) -> BoolArray:
    """Flat complete windows; callers may rule out windows by endpoint change.

    Nonfinite readings always invalidate a window. ``candidates`` is indexed
    by window start, and the returned mask has the same shape.
    """
    count = len(y) - window + 1
    result = np.zeros(count, dtype=bool)
    invalid = np.r_[0, np.cumsum(~(np.isfinite(x) & np.isfinite(y)))]
    eligible: BoolArray = candidates & (invalid[window:] == invalid[:-window])
    if not np.any(eligible) or math.isnan(alpha) or alpha > 1:
        return result
    if alpha <= 0:
        return eligible
    if window < 3 or alpha == 1 or alpha < 1e-10:
        for i in np.flatnonzero(eligible):
            result[i] = ols_slope(x[i : i + window], y[i : i + window])[2] >= alpha
        return result

    # Count exact changes, not a rounded variance: an exactly constant segment
    # can sit beside a very large excursion in the same prefix block.
    constant = _constant_windows(x, window) | _constant_windows(y, window)
    result[eligible & constant] = True
    eligible &= ~constant
    z = NormalDist().inv_cdf(alpha / 2)
    cutoff = z * z / (window - 2 + z * z)
    batch = max(window, 256)
    for lo in range(0, count, batch):
        hi = min(lo + batch, count)
        active = eligible[lo:hi]
        if not np.any(active):
            continue
        bx, by = x[lo : hi + window - 1], y[lo : hi + window - 1]
        # Invalid values cannot affect an eligible window. Zero them only in
        # this working copy, never in the input or its eligibility mask.
        with np.errstate(over="ignore", invalid="ignore"):
            dx = np.where(np.isfinite(bx), bx, 0.0)
            dy = np.where(np.isfinite(by), by, 0.0)
            dx -= dx[0]
            dy -= dy[0]
        sx_scale, sy_scale = np.max(np.abs(dx)), np.max(np.abs(dy))
        safe = bool(np.isfinite(sx_scale) and np.isfinite(sy_scale) and sx_scale > 0 and sy_scale > 0)
        trusted: BoolArray = np.zeros(hi - lo, dtype=bool)
        if safe:
            dx, dy = dx / sx_scale, dy / sy_scale
            dx -= dx.mean()
            dy -= dy.mean()
            sx, ex = _window_sums(dx, window)
            sy, ey = _window_sums(dy, window)
            xx, exx = _window_sums(dx * dx, window)
            yy, eyy = _window_sums(dy * dy, window)
            xy, exy = _window_sums(dx * dy, window)
            xx -= sx * sx / window
            yy -= sy * sy / window
            xy -= sx * sy / window
            err_xx = exx + (2 * np.abs(sx) * ex + ex * ex) / window
            err_yy = eyy + (2 * np.abs(sy) * ey + ey * ey) / window
            err_xy = exy + (np.abs(sx) * ey + np.abs(sy) * ex + ex * ey) / window
            delta = xy * xy - cutoff * xx * yy
            error = (2 * np.abs(xy) * err_xy + err_xy * err_xy) + cutoff * (
                np.abs(xx) * err_yy + np.abs(yy) * err_xx + err_xx * err_yy
            )
            # Include a margin for the direct fit and normal quantile rounding.
            error += 1e-10 * (xy * xy + cutoff * np.abs(xx * yy))
            trusted = active & (xx > err_xx) & (yy > err_yy) & (np.abs(delta) > error)
            result[lo:hi] |= trusted & (delta < 0)
        for offset in np.flatnonzero(active & ~trusted):
            i = lo + offset
            result[i] = ols_slope(x[i : i + window], y[i : i + window])[2] >= alpha
    return result
