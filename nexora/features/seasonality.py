"""Classical additive seasonal decomposition + period estimation."""
import math


def _clean_value(x):
    """Return float(x) or None if missing; raise ValueError if non-numeric."""
    if x is None:
        return None
    if isinstance(x, bool):
        raise ValueError(f"non-numeric entry: {x!r}")
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    raise ValueError(f"non-numeric entry: {x!r}")


def _popvar(vals):
    """Population variance of a pre-filtered float list; None if empty."""
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def decompose(values: list, period: int):
    """Additive decomposition x = trend + seasonal + residual.

    Trend = centered moving average of width `period` (even widths use
    2x averaging: half weight 0.5 on the two outermost points). Edges
    without a full window -> None. detrended = x - trend;
    seasonal[phase] = mean of detrended at indices i % period == phase;
    residual = x - trend - seasonal (None if any term missing).
    seasonal_strength = max(0, 1-var(resid)/var(detrended)),
    trend_strength = max(0, 1-var(resid)/var(deseasonalized)); 0.0 when
    a denominator is missing/zero. All three series match input length.
    Raises ValueError on bad period or non-numeric entries.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 2:
        raise ValueError("period must be an int >= 2")
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list/tuple")
    clean = [_clean_value(v) for v in values]
    n = len(clean)
    half = period // 2
    even = (period % 2 == 0)
    trend = [None] * n
    if period <= n:
        for i in range(n):
            lo = i - half
            hi = i + half
            if lo < 0 or hi >= n:
                continue
            if even:
                # window lo..hi inclusive (period+1 pts), 0.5 at ends.
                num, den = 0.0, 0.0
                for j in range(lo, hi + 1):
                    w = 0.5 if (j == lo or j == hi) else 1.0
                    if clean[j] is not None:
                        num += clean[j] * w
                        den += w
                trend[i] = num / den if den > 0 else None
            else:
                s, c = 0.0, 0
                for j in range(lo, hi + 1):
                    if clean[j] is not None:
                        s += clean[j]
                        c += 1
                trend[i] = s / c if c > 0 else None
    detrended = [clean[i] - trend[i] if clean[i] is not None
                 and trend[i] is not None else None for i in range(n)]
    phase_mean = []
    for p in range(period):
        vals = [detrended[i] for i in range(n)
                if i % period == p and detrended[i] is not None]
        phase_mean.append(sum(vals) / len(vals) if vals else None)
    seasonal = [phase_mean[i % period] for i in range(n)]
    residual = [clean[i] - trend[i] - seasonal[i]
                if clean[i] is not None and trend[i] is not None
                and seasonal[i] is not None else None
                for i in range(n)]
    resid = [v for v in residual if v is not None]
    detr = [v for v in detrended if v is not None]
    deseason = [clean[i] - seasonal[i] if clean[i] is not None
                and seasonal[i] is not None else None for i in range(n)]
    deseason = [v for v in deseason if v is not None]
    var_r, var_d, var_s = _popvar(resid), _popvar(detr), _popvar(deseason)
    if var_r is None or var_d is None or var_d == 0.0:
        seasonal_strength = 0.0
    else:
        seasonal_strength = max(0.0, 1.0 - var_r / var_d)
    if var_r is None or var_s is None or var_s == 0.0:
        trend_strength = 0.0
    else:
        trend_strength = max(0.0, 1.0 - var_r / var_s)
    return {"trend": trend, "seasonal": seasonal, "residual": residual,
            "period": period,
            "seasonal_strength": float(max(0.0, min(1.0, seasonal_strength))),
            "trend_strength": float(max(0.0, min(1.0, trend_strength))),
            "reason": f"additive MA({period}): seasonal_strength="
                      f"{seasonal_strength:.3f} trend_strength="
                      f"{trend_strength:.3f}"}


def _autocorr_global_mean(clean, mean, denom, lag):
    """Lag-k autocorrelation with global-mean estimator; 0.0 if undef."""
    if denom is None or denom == 0.0:
        return 0.0
    num = 0.0
    found = False
    for i in range(len(clean) - lag):
        if clean[i] is not None and clean[i + lag] is not None:
            num += (clean[i] - mean) * (clean[i + lag] - mean)
            found = True
    if not found:
        return 0.0
    r = num / denom
    return max(-1.0, min(1.0, r))


def estimate_period(values: list, min_period: int = 2, max_period=None):
    """Dominant period via autocorrelation peaks (self-contained).

    Selects the smallest lag k >= max(2, min_period) with r[k] > 0.3 that
    is a local maximum (r[k] > r[k-1] and r[k] >= r[k+1]; at max_period
    only the left comparison applies). Returns {"period" (int|None),
    "strength" (0..1), "method": "autocorr-peak", "reason"}.
    Deterministic. Raises ValueError on non-numeric entries.
    """
    if not isinstance(min_period, int) or isinstance(min_period, bool) \
            or min_period < 2:
        raise ValueError("min_period must be an int >= 2")
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list/tuple")
    clean = [_clean_value(v) for v in values]
    n = len(clean)
    valid = [v for v in clean if v is not None]
    if max_period is None:
        max_p = n // 2
    else:
        if not isinstance(max_period, int) or isinstance(max_period, bool):
            raise ValueError("max_period must be an int or None")
        max_p = max_period
    lower = max(2, min_period)
    if len(valid) < 3 or max_p < lower:
        return {"period": None, "strength": 0.0,
                "method": "autocorr-peak",
                "reason": "insufficient data or range for peak search"}
    max_p = min(max_p, n - 1)
    if max_p < lower:
        return {"period": None, "strength": 0.0,
                "method": "autocorr-peak",
                "reason": "insufficient data or range for peak search"}
    mean = sum(valid) / len(valid)
    denom = sum((v - mean) ** 2 for v in valid)
    if denom == 0.0:
        return {"period": None, "strength": 0.0,
                "method": "autocorr-peak",
                "reason": "zero variance: autocorrelation undefined"}
    rs = {}
    for k in range(max(0, lower - 1), max_p + 2):
        if 0 <= k < n:
            rs[k] = _autocorr_global_mean(clean, mean, denom, k)
    for k in range(lower, max_p + 1):
        r = rs.get(k, 0.0)
        if r <= 0.3:
            continue
        left = rs.get(k - 1, float("-inf"))
        if k < max_p:
            right = rs.get(k + 1, float("-inf"))
            is_peak = (r > left) and (r >= right)
        else:
            is_peak = (r > left)
        if is_peak:
            return {"period": k, "strength": float(r),
                    "method": "autocorr-peak",
                    "reason": f"smallest local autocorr peak > 0.3 at "
                              f"lag {k} (r={r:.3f})"}
    return {"period": None, "strength": 0.0, "method": "autocorr-peak",
            "reason": "no local autocorrelation peak > 0.3 found"}
